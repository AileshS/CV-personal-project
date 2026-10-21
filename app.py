"""A small review desk. Long-running work belongs to ``playlab worker``.

Only one requested frame is decoded here. Full-game importing, clipping,
indexing and model inference are queued; widget reruns never load models.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from playlab.analysis import analysis_is_fresh
from playlab.attention import resolve_attention
from playlab.box_drawer import draw_frame
from playlab.dataset import canonical_hash
from playlab.domain import (
    assign_game_split,
    assign_split_blocks,
    merge_possessions,
    save_possession,
    split_possession,
    validate_annotation,
)
from playlab.media import frame_at
from playlab.quick_analysis_ui import render_quick_analysis
from playlab.review import frame_selection, load_batch
from playlab.review_ui import render_review_batches
from playlab.store import Store

SPLITS = ["unassigned", "train", "val", "test"]
SPLIT_NAMES = {"unassigned": "Unassigned", "train": "Training", "val": "Validation", "test": "Final test"}


def clock(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, float(seconds)), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{remainder:05.2f}"


@st.cache_data(max_entries=48, show_spinner=False)
def review_frame(path: str, seconds: float, modification_time: float):
    """Cache only small stills, never the full source or a loaded detector."""
    return frame_at(path, seconds, max_width=1280)


def show_frame(game: dict, seconds: float, boxes=()):
    path = Path(game["source_path"])
    if not path.is_file():
        st.info("Source video is unavailable on this machine. Import or restore it to review frames.")
        return
    try:
        image = Image.fromarray(review_frame(str(path), seconds, path.stat().st_mtime)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item in boxes:
            box = item.get("box")
            if box and len(box) == 4:
                coords = [box[0] * image.width, box[1] * image.height,
                          box[2] * image.width, box[3] * image.height]
                color = ("#00d5ff" if item.get("draft") else "#ff8d54" if item.get("proposal") else
                         "#ffbd45" if item.get("label") == "ball" else "#17b7a2")
                draw.rectangle(coords, outline=color, width=3)
                name = item.get("label", "box")
                draw.text((coords[0], max(0, coords[1] - 15)),
                          f"Draft {name}" if item.get("draft") else name, fill=color)
        st.image(image, caption=f"Original-game time {clock(seconds)}",
                 alt="Basketball review frame with saved ball and rim boxes", width="stretch")
    except (OSError, RuntimeError, ValueError) as exc:
        st.warning(f"Could not decode this frame: {exc}")


def save_human(store: Store, kind: str, record: dict):
    record = validate_annotation(store, kind, {**record, "reviewed": True})
    store.save(kind, record)
    if kind == "shot":
        previous = store.get("analysis", record["possession_id"])
        if previous:
            store.save("analysis", {**previous, "stale": True})


def flash(message: str):
    st.session_state["notice"] = message
    st.rerun()


def box_inputs(prefix: str, value=None) -> list[float]:
    value = value or [0.4, 0.3, 0.5, 0.4]
    cols = st.columns(4)
    return [column.number_input(label, min_value=0.0, max_value=1.0,
                                value=float(coordinate), step=0.001, format="%.3f", key=f"{prefix}_{i}")
            for i, (column, label, coordinate) in enumerate(zip(cols, ["Left", "Top", "Right", "Bottom"], value))]


st.set_page_config(page_title="Basketball play lab", page_icon=":material/sports_basketball:", layout="wide")
st.title("Basketball play lab")
st.caption("Analyze a short interval and inspect the evidence. Detailed review remains available.")
store = Store(os.environ.get("PLAYLAB_DATA", "data"))
if "quick_imported_game" in st.session_state:
    st.session_state["game_choice"] = st.session_state.pop("quick_imported_game")

if "notice" in st.session_state:
    st.success(st.session_state.pop("notice"))

with st.sidebar:
    st.subheader("Review desk")
    view = st.selectbox("Workspace", ["Quick analysis", "Possessions", "Review batches", "Frame labels", "Game setup", "Import game", "Jobs and export"],
                        key="workspace")
    games = store.list("game")
    game_lookup = {g["id"]: g for g in games}
    selected_game = st.selectbox("Game", list(game_lookup), format_func=lambda key: game_lookup[key]["name"],
                                 key="game_choice") if games else None
    st.caption("Local viewer: keep a worker running separately for queued work.")
    if os.environ.get("PLAYLAB_NTFY_URL"):
        st.caption("A phone alert endpoint is set in this environment. Test delivery and phone receipt are still unverified.")
    else:
        st.caption("Phone alerts are not configured in this environment.")
    st.caption("A cloud worker needs its own alert settings. This local viewer alone cannot run with your computer off.")
    pending_attention = store.list("attention", resolved=False)
    if pending_attention:
        st.warning(f"{len(pending_attention)} action request(s). Open Jobs and export.")

game = game_lookup.get(selected_game)
possessions = sorted(store.list("possession", game_id=selected_game), key=lambda item: item["start"]) if game else []

if view == "Quick analysis":
    render_quick_analysis(store, game, clock)

elif view == "Import game":
    st.subheader("Import a game")
    st.write("Enter a full local video path or a YouTube URL. Import runs in the worker and keeps original timestamps.")
    with st.form("import_game"):
        source = st.text_input("Video path or YouTube URL", key="import_source")
        name = st.text_input("Game name", key="import_name")
        submitted = st.form_submit_button("Queue import", key="queue_import", type="primary")
    if submitted:
        if not source.strip():
            st.error("Enter a video path or URL.")
        elif not source.startswith(("https://", "http://")) and not Path(source).expanduser().is_file():
            st.error("That local video file does not exist.")
        else:
            job = store.enqueue("import", {"source": source.strip(), "name": name.strip() or "Basketball game"})
            st.success(f"Import queued ({job[:8]}). Open Jobs and export for progress.")
    st.info("Start the worker in a separate terminal: uv run playlab worker")

elif view == "Jobs and export":
    st.subheader("Jobs and export")
    st.subheader("Actions that need you")
    if not pending_attention:
        st.info("No unresolved action requests.")
    for request in pending_attention:
        with st.container(border=True):
            st.write(request["message"])
            delivery = request.get("notification", {})
            if delivery.get("delivered"):
                st.caption("Notification service accepted this alert. Receipt on your phone is not verified.")
            else:
                st.caption(delivery.get("reason", "No phone notification was delivered for this request."))
            if st.button("Mark action done", key=f"resolve_attention_{request['id']}"):
                resolve_attention(store, request["id"])
                flash("Action request marked done. Queued work still requires an active worker.")
    st.divider()
    st.button("Refresh job status", key="refresh_jobs")
    jobs = store.jobs()
    if not jobs:
        st.info("No queued jobs yet. Import a game to begin.")
    for job in jobs[:30]:
        with st.container(border=True):
            st.write(f"**{job['task'].capitalize()} · {job['status']}** · {job['id'][:8]}")
            st.progress(float(job["progress"]))
            if job.get("error"):
                st.error(job["error"])
            with st.container(horizontal=True):
                if job["status"] in {"queued", "running"} and st.button("Cancel", key=f"cancel_{job['id']}"):
                    store.cancel_job(job["id"])
                    flash("Cancellation requested. Running work stops at its next checkpoint.")
                if job["status"] in {"failed", "cancelled"} and st.button("Resume", key=f"resume_{job['id']}"):
                    try:
                        store.resume_job(job["id"])
                        flash("Job queued for resume from its saved checkpoint.")
                    except ValueError as exc:
                        st.error(str(exc))
    st.caption("After an interrupted worker, stop its old process and run the documented recovery command before resuming.")
    records = {kind: store.list(kind) for kind in
               ["game", "possession", "shot", "box", "frame", "event", "setup", "analysis", "attention",
                "review_batch", "review_item", "review_frame", "review_clip"]}
    st.download_button("Download review records", data=json.dumps(records, indent=2),
                       file_name="basketball-review.json", mime="application/json", key="export_records")
    st.caption("This download contains labels and predictions separately. Videos and model weights stay outside the export.")

elif view == "Review batches":
    render_review_batches(store, clock)

elif game is None:
    st.info("Start with Import game. Nothing has been trained or evaluated yet.")

elif view == "Game setup":
    st.subheader("Game setup")
    st.write("Assign dataset groups before labeling. Use separate games when available; use separated blocks only for a same-game pilot.")
    strategy = st.selectbox("Dataset strategy", ["Whole game", "Same-game pilot"],
                            index=int(game.get("split_mode") == "contiguous"), key=f"dataset_strategy_{game['id']}")
    if strategy == "Whole game":
        with st.form(f"game_split_{game['id']}"):
            split = st.selectbox("Assign entire game to", ["train", "val", "test"],
                                 format_func=lambda choice: SPLIT_NAMES[choice], key="whole_game_split")
            confirmed = st.checkbox("Apply this group to all possessions and their existing labels", key="confirm_game_split")
            assign = st.form_submit_button("Assign whole-game split", key="assign_game_split")
        if assign:
            if not confirmed:
                st.error("Confirm the whole-game assignment before changing existing labels.")
            else:
                assign_game_split(store, game["id"], split)
                # Clear pilot metadata when explicitly returning to game-level grouping.
                updated_game = store.get("game", game["id"])
                updated_game.pop("split_blocks", None)
                updated_game["split_mode"] = "game"
                store.save("game", updated_game)
                flash("Whole game and its labels assigned. Keep this game out of other dataset groups.")
    else:
        st.warning("Same-game results are a pilot. Keep a 10-second guard at internal split boundaries and group replays together.")
        blocks = game.get("split_blocks", [])
        defaults = [blocks[0]["end"], blocks[1]["end"]] if len(blocks) == 3 else [game["duration"] * .6, game["duration"] * .8]
        with st.form(f"pilot_split_{game['id']}"):
            first_cut = st.number_input("Training block ends (game seconds)", min_value=0.0,
                                        max_value=float(game["duration"]), value=float(defaults[0]), key="pilot_train_end")
            second_cut = st.number_input("Validation block ends (game seconds)", min_value=0.0,
                                         max_value=float(game["duration"]), value=float(defaults[1]), key="pilot_val_end")
            save_blocks = st.form_submit_button("Set pilot blocks before labeling", key="save_pilot_blocks")
        if save_blocks:
            has_labels = any(store.list(kind, game_id=game["id"]) for kind in ("shot", "box", "frame", "event"))
            if has_labels:
                st.error("Pilot blocks must be set before labeling. Existing labels require an explicit migration outside this form.")
            elif not 10 < first_cut < second_cut - 20 < game["duration"] - 30:
                st.error("Choose blocks with enough footage for the 10-second guards on both sides of internal boundaries.")
            else:
                blocks = [{"start": 0.0, "end": first_cut, "split": "train"},
                          {"start": first_cut, "end": second_cut, "split": "val"},
                          {"start": second_cut, "end": game["duration"], "split": "test"}]
                assign_split_blocks(store, game["id"], blocks)
                flash("Pilot blocks saved. Possessions touching a guard remain unassigned; keep them out of training exports.")
    st.divider()
    st.write("Match uniform colors and mark a rim visible in a representative frame. Revisit setup after a camera change.")
    setup = store.get("setup", game["id"]) or {}
    seconds = st.number_input("Setup frame time (game seconds)", min_value=0.0,
                              max_value=float(game["duration"]), value=float(setup.get("time", 0)),
                              step=0.1, key=f"setup_time_{game['id']}")
    show_frame(game, seconds, [{"label": "rim", "box": setup["rim"]}] if setup.get("rim") else [])
    with st.form(f"setup_{game['id']}"):
        colors = setup.get("team_colors", ["#ffffff", "#163b78"])
        left, right = st.columns(2)
        color_a = left.color_picker("Team A uniform", value=colors[0])
        color_b = right.color_picker("Team B uniform", value=colors[1])
        st.caption("Rim box: fractions of the frame, from 0 to 1. Left must be less than right; top less than bottom.")
        rim = box_inputs(f"setup_rim_{game['id']}", setup.get("rim"))
        reviewed = st.checkbox("I checked this setup against the frame", key=f"setup_reviewed_{game['id']}")
        save_setup = st.form_submit_button("Save setup", key="save_setup")
    if save_setup:
        if not reviewed:
            st.error("Review the frame before saving setup.")
        elif rim[0] >= rim[2] or rim[1] >= rim[3]:
            st.error("The rim needs positive width and height.")
        elif color_a == color_b:
            st.error("Choose different uniform colors for the two teams.")
        else:
            store.save("setup", {"id": game["id"], "game_id": game["id"], "time": seconds,
                                 "rim": rim, "team_colors": [color_a, color_b], "reviewed": True})
            for old_analysis in store.list("analysis", game_id=game["id"]):
                store.save("analysis", {**old_analysis, "stale": True})
            flash("Game setup saved. Validate it again when the camera view changes.")

else:
    st.subheader(game["name"])
    st.caption(f"{clock(game['duration'])} · {game['width']} × {game['height']} · Source timestamps retained")
    if view == "Possessions":
        with st.container(horizontal=True):
            if st.button("Propose possessions", key="queue_index"):
                store.enqueue("index", {"game_id": game["id"]})
                flash("Possession proposal job queued. Automatic boundaries need review.")
        with st.expander("Add a possession manually", expanded=not possessions):
            with st.form(f"new_possession_{game['id']}"):
                start = st.number_input("Start (game seconds)", min_value=0.0,
                                        max_value=float(game["duration"]), value=0.0, key="new_start")
                end = st.number_input("End (game seconds)", min_value=0.0,
                                      max_value=float(game["duration"]), value=min(24.0, game["duration"]), key="new_end")
                create = st.form_submit_button("Add possession", key="add_possession")
            if create:
                try:
                    save_possession(store, game["id"], start, end)
                    flash("Possession added. Assign a dataset split and review its boundaries.")
                except ValueError as exc:
                    st.error(str(exc))
    if not possessions:
        st.info("No possessions yet. Propose boundaries or add a reviewed segment manually.")
    else:
        lookup = {p["id"]: p for p in possessions}
        possession_id = st.selectbox("Possession", list(lookup), key=f"possession_choice_{game['id']}",
                                    format_func=lambda key: f"{clock(lookup[key]['start'])}–{clock(lookup[key]['end'])} · "
                                                            f"{lookup[key]['split']} · "
                                                            f"{'reviewed' if lookup[key]['reviewed'] else 'needs review'}")
        possession = lookup[possession_id]
        prefix = f"{possession_id}_{possession['revision']}"
        if view == "Frame labels":
            st.write("Choose a frame, mark every visible ball and rim, then mark the frame reviewed. Negative frames matter too.")
            seconds = st.number_input("Frame time (game seconds)", min_value=float(possession["start"]),
                                      max_value=float(possession["end"]), value=float(possession["start"]),
                                      step=0.1, format="%.3f", key=f"frame_time_{prefix}")
            def same_frame(item):
                return abs(float(item["time"]) - seconds) < 0.0005

            boxes = [b for b in store.list("box", possession_id=possession_id) if same_frame(b)]
            frame_id = f"{possession_id}:{seconds:.3f}"
            reviewed_frame = store.get("frame", frame_id)
            image_slot = st.container()
            input_mode = st.radio("Box input", ["Click on frame", "Coordinate fields"], horizontal=True,
                                  key="box_input_mode")
            st.caption(f"Saved boxes: {len(boxes)} · Frame {'reviewed' if reviewed_frame and reviewed_frame['reviewed'] else 'not reviewed'}")
            box_lookup = {b["id"]: b for b in boxes}
            box_id = st.selectbox("Box to edit", ["new", *box_lookup], key=f"box_choice_{frame_id}",
                                  format_func=lambda key: "New box" if key == "new" else f"{box_lookup[key]['label']} · {key[:8]}")
            old_box = box_lookup.get(box_id, {})
            draft_key = f"box_draft_saved_{frame_id}_{box_id}"
            drawn_key = f"box_draft_drawn_{frame_id}_{box_id}"
            pending_key = f"box_pending_proposal_{frame_id}_{box_id}"
            reload_key = f"box_draft_reload_{frame_id}_{box_id}"
            if st.session_state.pop(reload_key, False):
                fresh = {"label": old_box.get("label", "ball"), "box": old_box.get("box", [.4, .3, .5, .4])}
                st.session_state[draft_key] = fresh
                st.session_state[drawn_key] = False
                st.session_state.pop(pending_key, None)
                st.session_state[f"box_{frame_id}_{box_id}_label"] = fresh["label"]
                for i, value in enumerate(fresh["box"]):
                    st.session_state[f"box_{frame_id}_{box_id}_{i}"] = float(value)
            label = st.selectbox("Object", ["ball", "rim"], index=1 if old_box.get("label") == "rim" else 0,
                                 key=f"box_{frame_id}_{box_id}_label")
            st.caption("Your unsaved box is cyan. Select Ball or Rim, then click two opposite corners around it. "
                       "Save reviewed box keeps it; drawing alone saves no label.")
            with st.expander("Fine-tune coordinates", expanded=input_mode == "Coordinate fields"):
                coordinates = box_inputs(f"box_{frame_id}_{box_id}", old_box.get("box"))
            valid_draft = (all(math.isfinite(v) and 0 <= v <= 1 for v in coordinates)
                           and coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3])
            baseline = st.session_state.setdefault(draft_key, {
                "label": old_box.get("label", "ball"), "box": old_box.get("box", [.4, .3, .5, .4])})
            draft_changed = (label != baseline["label"] or coordinates != baseline["box"]
                             or st.session_state.get(drawn_key, False))

            def discard_draft():
                st.session_state[drawn_key] = False
                st.session_state.pop(pending_key, None)
                st.session_state[f"box_{frame_id}_{box_id}_label"] = baseline["label"]
                for i, value in enumerate(baseline["box"]):
                    st.session_state[f"box_{frame_id}_{box_id}_{i}"] = float(value)

            preview_boxes = [b for b in boxes if b["id"] != box_id]
            has_visible_draft = (input_mode == "Coordinate fields" or box_id != "new" or draft_changed
                                 or st.session_state.get(drawn_key, False))
            if valid_draft and has_visible_draft:
                preview_boxes.append({"label": label, "box": coordinates, "draft": True})
            elif not valid_draft:
                st.warning("The draft box needs Left < Right and Top < Bottom. It has not been saved.")
            if context := st.session_state.get("review_frame_selection"):
                try:
                    shown_batch = load_batch(context["path"])
                    shown_selection = frame_selection(store, shown_batch, context["frame_id"])
                    if shown_selection["possession_id"] == possession_id and abs(shown_selection["time"] - seconds) < .0005:
                        for index, suggestion in enumerate(shown_batch.frames[context["frame_id"]].get("proposed_boxes", [])):
                            identifier = canonical_hash([shown_batch.id, context["frame_id"], possession_id, index])[:24]
                            if not store.get("proposal_decision", identifier):
                                preview_boxes.append({"label": f"{index + 1}: {suggestion['label']}",
                                                      "box": suggestion["box"], "proposal": True})
                except (ValueError, OSError, KeyError):
                    pass  # The link error is explained beside the review controls below.
            with image_slot:
                if input_mode == "Coordinate fields":
                    show_frame(game, seconds, preview_boxes)
                else:
                    image_path = Path(game["source_path"])
                    if image_path.is_file():
                        drawer_key = f"draw_box_{frame_id}_{box_id}"

                        def take_drawn_box():
                            state = st.session_state[drawer_key]
                            drawn = state.get("box") if isinstance(state, dict) else state.box
                            if (isinstance(drawn, (list, tuple)) and len(drawn) == 4
                                    and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in drawn)
                                    and drawn[0] < drawn[2] and drawn[1] < drawn[3]):
                                for i, value in enumerate(drawn):
                                    st.session_state[f"box_{frame_id}_{box_id}_{i}"] = float(value)
                                st.session_state[drawn_key] = True

                        draw_frame(review_frame(str(image_path), seconds, image_path.stat().st_mtime),
                                   preview_boxes, identity=f"{frame_id}:{box_id}:{label}",
                                   key=drawer_key, on_box_change=take_drawn_box)
                    else:
                        st.info("Restore the original source video to draw on this frame.")
            save_box = st.button("Save reviewed box", key="save_box", disabled=not valid_draft or not has_visible_draft)
            if draft_changed:
                st.info("Save this box or discard its changes before reviewing the complete frame. "
                        "Moving the preview does not save a label.")
                st.button("Discard box changes", key="discard_box_draft", on_click=discard_draft)
            # Keep the revisions that produced the displayed editor across the submit rerun.
            edit_key = f"{game['id']}:{possession_id}:{seconds:.3f}:{box_id}"
            current_snapshot = {
                "checks": [("possession", possession_id, possession["revision"]),
                           ("game", game["id"], game["revision"])],
                "frame_revision": reviewed_frame["revision"] if reviewed_frame else 0,
                "box_revision": old_box.get("revision", 0),
                "boxes": {b["id"]: b["revision"] for b in store.list("box", possession_id=possession_id)},
            }
            linked_error = None
            if context := st.session_state.get("review_frame_selection"):
                try:
                    selection = frame_selection(store, load_batch(context["path"]), context["frame_id"])
                    if selection["possession_id"] == possession_id:
                        current_snapshot["checks"].append((selection.get("review_kind", "review_item"), selection["review_item_id"],
                                                           selection["review_revision"]))
                except (ValueError, OSError) as exc:
                    linked_error = str(exc)
            snapshots = st.session_state.setdefault("frame_edit_snapshots", {})
            snapshot = snapshots.setdefault(edit_key, current_snapshot)
            if st.button("Reload frame editor", key="reload_frame_editor"):
                snapshots[edit_key] = current_snapshot
                st.session_state[reload_key] = True
                st.rerun()
            if linked_error:
                st.warning(f"Review link needs reconfirmation: {linked_error}")

            def write_frame(changes=(), deletes=()):
                if linked_error:
                    raise ValueError("Review link changed. Return to Review batches and explicitly reconfirm it.")
                store.save_many(changes, checks=[*snapshot["checks"], ("frame", frame_id, snapshot["frame_revision"])],
                                deletes=deletes,
                                collections=[("box", {"possession_id": possession_id}, snapshot["boxes"])])
                snapshots.pop(edit_key, None)

            if context := st.session_state.get("review_frame_selection"):
                try:
                    proposal_batch = load_batch(context["path"])
                    proposal_selection = frame_selection(store, proposal_batch, context["frame_id"])
                    if (proposal_selection["possession_id"] == possession_id
                            and abs(proposal_selection["time"] - seconds) < .0005):
                        suggestions = proposal_batch.frames[context["frame_id"]].get("proposed_boxes", [])
                        st.subheader("Check model suggestions")
                        st.caption("Accept only a correctly placed box. Reject records a bad suggestion; it does not mark a negative frame. "
                                   "Add missing objects with the click tool before reviewing the complete frame.")
                        if not suggestions:
                            st.info("No model suggestions at this timestamp. Use the click tool to add visible objects.")
                        for index, suggestion in enumerate(suggestions):
                            identifier = canonical_hash([proposal_batch.id, context["frame_id"], possession_id, index])[:24]
                            accepted_id = f"suggested_{identifier}"
                            decision = store.get("proposal_decision", identifier)
                            status = (decision or {}).get("decision", "unreviewed")
                            st.caption(f"Suggestion {index + 1}: {suggestion['label']} · {suggestion['confidence']:.0%} · {status}")
                            controls = st.columns(3)
                            accepted = controls[0].button("👍 Accept box", key=f"accept_{identifier}",
                                disabled=draft_changed or store.get("box", accepted_id) is not None)
                            rejected = controls[1].button("👎 Reject box", key=f"reject_{identifier}",
                                disabled=draft_changed or store.get("box", accepted_id) is not None or status == "rejected")

                            pending = {"id": identifier, "batch_hash": proposal_batch.id,
                                       "frame_id": context["frame_id"], "proposal_index": index,
                                       "decision_revision": (decision or {}).get("revision", 0)}

                            def correct_suggestion(item=suggestion, correction=pending):
                                st.session_state[f"box_choice_{frame_id}"] = "new"
                                st.session_state[f"box_{frame_id}_new_label"] = item["label"]
                                st.session_state[f"box_draft_drawn_{frame_id}_new"] = True
                                st.session_state[f"box_pending_proposal_{frame_id}_new"] = correction
                                for i, value in enumerate(item["box"]):
                                    st.session_state[f"box_{frame_id}_new_{i}"] = float(value)

                            controls[2].button("Correct box", key=f"correct_{identifier}", on_click=correct_suggestion,
                                               disabled=store.get("box", accepted_id) is not None)
                            if accepted or rejected:
                                action = {"id": identifier, "batch_hash": proposal_batch.id,
                                          "frame_id": context["frame_id"], "game_id": game["id"],
                                          "possession_id": possession_id, "time": seconds, "proposal_index": index,
                                          "decision": "accepted" if accepted else "rejected", "source": "human"}
                                changes = [("proposal_decision", action, (decision or {}).get("revision", 0))]
                                if accepted:
                                    box = validate_annotation(store, "box", {
                                        "id": accepted_id, "game_id": game["id"], "possession_id": possession_id,
                                        "time": seconds, "split": possession["split"], "label": suggestion["label"],
                                        "box": suggestion["box"], "reviewed": True, "proposal_ref": identifier})
                                    changes.append(("box", box, 0))
                                    if reviewed_frame:
                                        changes.append(("frame", {**reviewed_frame, "reviewed": False}, snapshot["frame_revision"]))
                                try:
                                    if draft_changed:
                                        raise ValueError("Save or discard the current draft first.")
                                    write_frame(changes)
                                    flash("Suggestion accepted. Check remaining objects before reviewing the frame." if accepted else
                                          "Suggestion rejected. Other objects may still need labels.")
                                except ValueError as exc:
                                    st.error(str(exc))
                    else:
                        st.caption("Model suggestions apply to the batch's exact sampled timestamp. Reopen it from Review batches.")
                except (ValueError, OSError, KeyError) as exc:
                    st.info(f"Model suggestion review unavailable: {exc}")

            remove_review = delete_selected = complete = negative = False
            with st.container(horizontal=True):
                if reviewed_frame:
                    remove_review = st.button("Remove frame review label", key="remove_frame_review")
                if old_box:
                    delete_selected = st.button("Delete selected box", key="delete_box")
                complete = st.button("Mark complete frame reviewed", key="review_frame", disabled=not boxes or draft_changed)
                negative = st.button("Reviewed negative: no ball or rim", key="negative_frame", disabled=bool(boxes) or draft_changed)
            try:
                if save_box:
                    if not valid_draft or not has_visible_draft:
                        raise ValueError("Draw a valid visible box before saving.")
                    record = {"game_id": game["id"], "possession_id": possession_id, "time": seconds,
                              "label": label, "box": coordinates, "split": possession["split"]}
                    if old_box:
                        record["id"] = old_box["id"]
                    pending = st.session_state.get(pending_key)
                    if pending and not old_box:
                        record.update(id=f"suggested_{pending['id']}", proposal_ref=pending["id"])
                    record = validate_annotation(store, "box", {**record, "reviewed": True})
                    changes = [("box", record, snapshot["box_revision"])]
                    if pending and not old_box:
                        decision = {k: v for k, v in pending.items() if k != "decision_revision"}
                        decision.update(game_id=game["id"], possession_id=possession_id,
                                        time=seconds, source="human", decision="corrected")
                        changes.append(("proposal_decision", decision, pending["decision_revision"]))
                    if reviewed_frame:
                        changes.append(("frame", {**reviewed_frame, "reviewed": False, "negative": False},
                                        snapshot["frame_revision"]))
                    write_frame(changes)
                    st.session_state[draft_key] = {"label": label, "box": list(coordinates)}
                    st.session_state[drawn_key] = False
                    st.session_state.pop(pending_key, None)
                    flash("Box saved. Mark the complete frame reviewed when all boxes are correct.")
                if remove_review:
                    write_frame(deletes=[("frame", frame_id, snapshot["frame_revision"])])
                    flash("Frame review removed. Add or correct boxes, then review the frame again.")
                if delete_selected:
                    changes = []
                    if reviewed_frame:
                        changes.append(("frame", {**reviewed_frame, "reviewed": False}, snapshot["frame_revision"]))
                    write_frame(changes, deletes=[("box", old_box["id"], snapshot["box_revision"])])
                    flash("Box deleted. Review the frame again before exporting training data.")
                if complete or negative:
                    if draft_changed:
                        raise ValueError("Save or discard the current draft before reviewing the frame.")
                    record = validate_annotation(store, "frame", {"id": frame_id, "game_id": game["id"],
                        "possession_id": possession_id, "time": seconds, "negative": negative,
                        "split": possession["split"], "reviewed": True})
                    write_frame([("frame", record, snapshot["frame_revision"])])
                    flash("Negative frame reviewed. It will help measure false detections." if negative else
                          "Frame reviewed with its saved boxes.")
            except ValueError as exc:
                st.error(str(exc))
        else:
            clip = store.get("clip", possession_id)
            if (clip and not clip.get("stale") and clip.get("possession_revision") == possession["revision"]
                    and Path(clip["path"]).is_file()):
                st.video(clip["path"], alt="Selected basketball possession")
            else:
                st.info("Prepare a short play clip to watch this possession without loading the full game.")
                if st.button("Prepare play clip", key="queue_clip"):
                    store.enqueue("clip", {"possession_id": possession_id})
                    flash("Play clip queued. Refresh when the worker finishes.")
            with st.expander("Edit possession and dataset split"):
                with st.form(f"edit_{prefix}"):
                    start = st.number_input("Start (game seconds)", min_value=0.0, max_value=float(game["duration"]),
                                            value=float(possession["start"]), step=0.1, key=f"edit_start_{prefix}")
                    end = st.number_input("End (game seconds)", min_value=0.0, max_value=float(game["duration"]),
                                          value=float(possession["end"]), step=0.1, key=f"edit_end_{prefix}")
                    split = st.selectbox("Dataset split", SPLITS, index=SPLITS.index(possession["split"]),
                                         format_func=lambda choice: SPLIT_NAMES[choice], key=f"split_{prefix}")
                    checked = st.checkbox("I reviewed the whole possession boundaries", value=possession["reviewed"], key=f"checked_{prefix}")
                    st.caption("Keep related frames, shots and replays in one split. Final test footage must stay untouched during model selection.")
                    update = st.form_submit_button("Save possession", key="save_possession")
                if update:
                    try:
                        save_possession(store, game["id"], start, end, record_id=possession_id, reviewed=checked, split=split)
                        flash("Possession saved. Cached clips and predictions may need refreshing after a boundary change.")
                    except ValueError as exc:
                        st.error(str(exc))
                with st.form(f"split_possession_{prefix}"):
                    at = st.number_input("Split at (game seconds)", min_value=float(possession["start"]),
                                         max_value=float(possession["end"]), value=(possession["start"] + possession["end"]) / 2)
                    divide = st.form_submit_button("Split possession", key="split_possession")
                if divide:
                    try:
                        split_possession(store, possession_id, at)
                        flash("Possession split. Check both resulting segments.")
                    except ValueError as exc:
                        st.error(str(exc))
                position = list(lookup).index(possession_id)
                if position + 1 < len(possessions) and st.button("Merge with next possession", key="merge_next"):
                    try:
                        merge_possessions(store, [possession_id, possessions[position + 1]["id"]])
                        flash("Possessions merged. Check the combined boundaries.")
                    except ValueError as exc:
                        st.error(str(exc))
            st.subheader("Analyze this possession")
            with st.form(f"analyze_{prefix}"):
                device = st.selectbox("Processing device", ["cpu", "cuda:0"], key="device")
                mode = st.selectbox("Release mode", ["assisted", "automatic"], key="analysis_mode",
                                    format_func=lambda value: "Use reviewed release anchors" if value == "assisted"
                                    else "Discover shots automatically",
                                    help="Automatic analysis ignores saved shot anchors. Evaluate each mode separately.")
                weights = st.text_input("Ball/rim model path (optional)", key="model_weights",
                                        help="A reviewed fine-tuned detector improves ball and rim detection. Blank uses the documented baseline.")
                classifier = st.text_input("Shot-classifier bundle path (optional)", key="classifier_bundle",
                                            help="Use the bundle saved by the classifier training command. Blank uses the trajectory baseline.")
                analyze = st.form_submit_button("Queue analysis", key="queue_analysis", type="primary")
            if analyze:
                payload = {"possession_id": possession_id, "device": device, "mode": mode}
                if weights.strip():
                    payload["weights"] = weights.strip()
                if classifier.strip():
                    payload["model_bundle"] = classifier.strip()
                store.enqueue("analyze", payload)
                flash("Analysis queued. Predictions remain separate from your labels.")
            analyses = store.list("analysis", possession_id=possession_id)
            analyses = [item for item in analyses if item.get("analysis_mode", "assisted") == mode]
            if analyses:
                analysis = analyses[-1]
                stale = not analysis_is_fresh(store, analysis)
                if stale:
                    st.warning("Possession boundaries, shot anchors or setup changed after this analysis. Run it again.")
                st.warning("Experimental results: check evidence. Closest defender means approximate image proximity, not court distance.")
                for warning in analysis.get("warnings", []):
                    st.caption(warning)
                if analysis.get("status"):
                    st.write(f"Processing status: {analysis['status']}. Human review is still required.")
                for i, event in enumerate(analysis.get("events", []), 1):
                    with st.container(border=True):
                        st.write(f"**Attempt {i}: {event.get('result', 'unknown')}** · {event.get('status', 'needs review')}")
                        if event.get("release") is not None:
                            st.write(f"Release: {clock(event['release'])}")
                        if event.get("defender"):
                            nearest = event["defender"]
                            if nearest.get("track_id") is not None:
                                st.write(f"Closest visible opponent: track {nearest['track_id']} (approximate image proximity)")
                            else:
                                st.caption(f"Defender: {nearest.get('reason', 'needs review')}")
                        evidence = event.get("evidence_path")
                        if evidence and Path(evidence).is_file():
                            st.image(evidence, alt="Shot evidence at release with player highlights", width="stretch")
                        if event.get("reason"):
                            st.caption(str(event["reason"]))
                if not analysis.get("events"):
                    st.info("No supported shot attempts found. This does not prove the possession had no shot.")
                with st.expander("Automatic evidence record"):
                    st.json(analysis)
            else:
                st.info("No analysis yet. You can label this possession manually while collecting training data.")
            st.subheader("Your reviewed labels")
            shots = store.list("shot", possession_id=possession_id)
            shot_lookup = {s["id"]: s for s in shots}
            shot_id = st.selectbox("Shot to label", ["new", *shot_lookup], key=f"shot_choice_{prefix}",
                                   format_func=lambda key: "New shot" if key == "new" else f"{clock(shot_lookup[key]['release'])} · {shot_lookup[key]['result']}")
            shot = shot_lookup.get(shot_id, {})
            with st.form(f"shot_{prefix}_{shot_id}"):
                release = st.number_input("Release time (game seconds)", min_value=float(possession["start"]),
                                          max_value=float(possession["end"]), value=float(shot.get("release", possession["start"])), step=0.01)
                result = st.selectbox("Shot result", ["unknown", "made", "missed"], index=["unknown", "made", "missed"].index(shot.get("result", "unknown")))
                shooter = st.text_input("Shooter track ID (blank if unclear)", value=shot.get("shooter_id") or "")
                defender = st.text_input("Closest opponent track ID (blank if unclear)", value=shot.get("defender_id") or "")
                free_throw = st.checkbox("Free throw", value=bool(shot.get("free_throw", False)))
                confirmed = st.checkbox("I checked the release and outcome", key=f"confirm_shot_{prefix}_{shot_id}")
                save_shot = st.form_submit_button("Save reviewed shot", key="save_shot")
            if save_shot:
                if not confirmed:
                    st.error("Review the attempt before saving a verified label.")
                else:
                    try:
                        record = {"game_id": game["id"], "possession_id": possession_id, "release": release,
                                  "result": result, "shooter_id": shooter.strip() or None,
                                  "defender_id": None if free_throw else defender.strip() or None,
                                  "free_throw": free_throw, "shot_type": "free_throw" if free_throw else "field_goal",
                                  "split": possession["split"]}
                        if shot:
                            record["id"] = shot["id"]
                        save_human(store, "shot", record)
                        flash("Reviewed shot saved separately from automatic predictions.")
                    except ValueError as exc:
                        st.error(str(exc))
            if shot and st.button("Delete selected shot label", key="delete_shot"):
                store.delete("shot", shot["id"])
                previous = store.get("analysis", possession_id)
                if previous:
                    store.save("analysis", {**previous, "stale": True})
                flash("Human shot label deleted. Automatic evidence retained.")
            with st.expander("Manual foul or turnover label"):
                st.caption("These labels build future training data. Automatic foul and turnover prediction is deferred.")
                with st.form(f"event_{prefix}"):
                    event_type = st.selectbox("Event", ["foul", "turnover"])
                    event_time = st.number_input("Event time (game seconds)", min_value=float(possession["start"]),
                                                 max_value=float(possession["end"]), value=float(possession["start"]), step=0.1)
                    notes = st.text_input("Notes", key=f"event_notes_{prefix}")
                    verified_event = st.checkbox("I verified this event", key=f"event_reviewed_{prefix}")
                    save_event = st.form_submit_button("Save manual event", key="save_event")
                if save_event:
                    if not verified_event:
                        st.error("Verify the event before saving.")
                    else:
                        try:
                            save_human(store, "event", {"game_id": game["id"], "possession_id": possession_id,
                                                       "time": event_time, "type": event_type, "notes": notes,
                                                       "split": possession["split"]})
                            flash("Manual event saved for future training.")
                        except ValueError as exc:
                            st.error(str(exc))
                for saved_event in store.list("event", possession_id=possession_id):
                    st.write(f"{clock(saved_event['time'])} · {saved_event['type']} · {saved_event.get('notes', '')}")
                    if st.button("Delete event label", key=f"delete_event_{saved_event['id']}"):
                        store.delete("event", saved_event["id"])
                        flash("Manual event label deleted.")
