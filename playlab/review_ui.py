"""Streamlit review-batch browser; suggestions never become annotations here."""
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from playlab.review import (
    confirm_possession,
    frame_review,
    frame_selection,
    load_batch,
    match_game,
    original_available,
    register_batch,
    request_missing_media,
    save_frame_review,
    save_navigation,
)
from playlab.review_clips import get_review_clip, queue_review_clip


def render_review_batches(store, clock):
    st.subheader("Review batches")
    st.caption("Sampling windows are not verified possessions. Suggestions remain unreviewed; navigation saves no labels.")
    saved = store.list("review_batch")
    default = saved[-1]["path"] if saved else ""
    path = st.text_input("Local manifest path", value=default, key="review_manifest_path")
    if st.button("Load review batch", key="load_review_batch"):
        try:
            batch = load_batch(Path(path).expanduser())
            record = register_batch(store, path)
            st.session_state["review_active_path"] = path
            st.session_state["review_nav"] = record
            st.session_state.pop("review_window", None)
            st.session_state.pop("review_frame", None)
            st.session_state.pop("review_snapshot", None)
        except (ValueError, OSError, KeyError) as exc:
            st.error(str(exc))
    active = st.session_state.get("review_active_path", default)
    if not active:
        st.info("Load a versioned manifest.json from this machine. A preview-only archive can be browsed without importing a game.")
        return
    try:
        batch = load_batch(active)
        nav = st.session_state.setdefault("review_nav", register_batch(store, active))
        frames = batch.frames
        windows = list(batch.items)
        if not windows or not frames:
            st.info("This batch has no review frames.")
            return
        current = nav.get("current_frame_id") or next(iter(frames))
        if current not in frames:
            current = next(iter(frames))

        def move(frame_id):
            try:
                st.session_state["review_nav"] = save_navigation(
                    store, batch, frame_id, expected_revision=st.session_state["review_nav"]["revision"])
                st.session_state["review_window"] = frames[frame_id]["review_item_id"]
                st.session_state["review_frame"] = frame_id
            except ValueError as exc:
                st.session_state["review_navigation_error"] = str(exc)

        def choose_window():
            move(next(fid for fid, frame in frames.items()
                      if frame["review_item_id"] == st.session_state["review_window"]))

        def choose_frame():
            move(st.session_state["review_frame"])

        if error := st.session_state.pop("review_navigation_error", None):
            st.error(error)
        window_id = frames[current]["review_item_id"]
        st.session_state.setdefault("review_window", window_id)
        st.selectbox("Review window", windows, key="review_window", on_change=choose_window,
                     format_func=lambda key: f"{windows.index(key) + 1} · {clock(batch.items[key]['start'])}–"
                                             f"{clock(batch.items[key]['end'])} · {batch.items[key]['split']}")
        controls = st.columns(2)
        position = windows.index(window_id)
        def window_first(index):
            return next(fid for fid, frame in frames.items() if frame["review_item_id"] == windows[index])

        controls[0].button("Previous window", key="review_previous_window", disabled=position == 0,
                           on_click=move, args=(window_first(max(0, position - 1)),))
        controls[1].button("Next window", key="review_next_window", disabled=position == len(windows) - 1,
                           on_click=move, args=(window_first(min(len(windows) - 1, position + 1)),))
        window_frames = [fid for fid, frame in frames.items() if frame["review_item_id"] == window_id]
        st.session_state.setdefault("review_frame", current)
        st.selectbox("Review frame", window_frames, key="review_frame", on_change=choose_frame,
                     format_func=lambda key: f"{clock(frames[key]['time'])} · {frames[key]['split']}")
        controls = st.columns(2)
        position = window_frames.index(current)
        controls[0].button("Previous frame", key="review_previous_frame", disabled=position == 0,
                           on_click=move, args=(window_frames[max(0, position - 1)],))
        controls[1].button("Next frame", key="review_next_frame", disabled=position == len(window_frames) - 1,
                           on_click=move, args=(window_frames[min(len(window_frames) - 1, position + 1)],))
        frame, item = frames[current], batch.items[window_id]
        st.caption(f"Original timestamp {clock(frame['time'])} ({frame['time']:.3f} seconds) · "
                   f"Split {frame['split']} · Source {frame['game_id']} · Batch {batch.manifest['batch_hash'][:12]}")
        media = batch.media(current)
        if media["native_path"]:
            with Image.open(media["native_path"]) as source_image:
                native_image = source_image.convert("RGB")
            if st.checkbox("Show unreviewed suggested boxes on native frame", value=False, key="review_show_suggestions"):
                draw = ImageDraw.Draw(native_image)
                for proposal in frame.get("proposed_boxes", []):
                    box = proposal.get("box", [])
                    if len(box) == 4:
                        coordinates = [box[0] * native_image.width, box[1] * native_image.height,
                                       box[2] * native_image.width, box[3] * native_image.height]
                        draw.rectangle(coordinates, outline="#ffbd45", width=3)
                        draw.text((coordinates[0], max(0, coordinates[1] - 15)),
                                  f"unreviewed {proposal.get('label', 'box')}", fill="#ffbd45")
            st.image(native_image, caption=f"Native frame · {frame['width']} × {frame['height']}", width="stretch")
        elif media["preview_path"]:
            st.image(str(media["preview_path"]), caption="Preview thumbnail with unreviewed suggested overlays; native image missing",
                     width="stretch")
        else:
            st.warning("Neither native frame nor preview is available.")
        st.subheader("Unreviewed model suggestions")
        st.caption("Reference only. Missing suggestions are not verified negatives. Correct every visible ball/rim in Frame labels.")
        if frame.get("proposed_boxes"):
            st.json(frame["proposed_boxes"])
        else:
            st.info("No suggestions for this frame; it remains unreviewed.")
        game = None
        try:
            game = match_game(store, batch, frame["game_id"])
            available = bool(media["native_path"]) and original_available(store, batch, frame["game_id"])
        except ValueError:
            available = False
        if not available:
            request_missing_media(store, batch, current)
            st.warning("Browse only: restore the original matching video and native frames on this machine. "
                       "Import the original game if absent. Thumbnails cannot supply training labels or verify possession boundaries. "
                       "A persistent action request is available in Jobs and export.")
        if available and game:
            st.caption(f"Original-video context {clock(item['start'])}–{clock(item['end'])}. "
                       "Prepare this bounded review window in the worker, then inspect the whole possession before confirming.")
            context_columns = st.columns(2)
            context_start = context_columns[0].number_input("Video context start (source seconds)", min_value=0.0,
                max_value=float(game['duration']), value=float(item['start']), key=f"review_context_start_{window_id}")
            context_end = context_columns[1].number_input("Video context end (source seconds)", min_value=0.0,
                max_value=float(game['duration']), value=float(item['end']), key=f"review_context_end_{window_id}")
            st.caption("Context clips are limited to 120 seconds and must stay within the declared training/validation partition.")
            try:
                clip = get_review_clip(store, batch, current, start=context_start, end=context_end)
                if clip and clip.get("status") == "ready" and clip.get("path"):
                    st.video(clip["path"])
                    st.caption(f"Clip starts at original source time {clock(clip['start'])}; playback time is relative to this clip.")
                else:
                    if clip:
                        st.info(f"Review-window clip: {clip['status']}. Keep a worker running and refresh to check progress.")
                    if st.button("Prepare short review-window clip", key="queue_review_clip"):
                        queue_review_clip(store, batch, current, start=context_start, end=context_end)
                        st.rerun()
                    st.button("Refresh review-window clip", key="refresh_review_clip")
            except ValueError as exc:
                st.info(f"Short video context unavailable: {exc}")
        snapshot_key = f"{batch.manifest['batch_hash']}:{window_id}"
        decisions = st.session_state.setdefault("review_decision_snapshots", {})
        decision = decisions.setdefault(f"{snapshot_key}:{current}", frame_review(store, batch, current))
        if st.button("Reload saved review", key="reload_review"):
            refreshed = frame_review(store, batch, current)
            refreshed_target = store.get("possession", refreshed.get("possession_id", ""))
            if refreshed_target and refreshed_target['revision'] != refreshed.get("possession_revision"):
                refreshed = {**refreshed, "draft_start": refreshed_target['start'], "draft_end": refreshed_target['end'],
                             "draft_split": refreshed_target['split'], "draft_target_id": refreshed_target['id'],
                             "draft_target_revision": refreshed_target['revision']}
            decisions[f"{snapshot_key}:{current}"] = refreshed
            st.session_state.pop("review_candidate_snapshots", None)
            for widget_key in list(st.session_state):
                if widget_key.startswith(f"review_{snapshot_key}_{current}_"):
                    st.session_state.pop(widget_key, None)
            st.rerun()
        prefix = f"review_{snapshot_key}_{current}_{decision['revision']}"
        st.subheader("Human boundary correction and notes")
        st.caption("Review the entire offensive possession, retaining offensive rebounds. Explicitly confirm the split before linking frame labels.")
        target = store.get("possession", decision.get("possession_id", ""))
        if target:
            def edit_target():
                st.session_state["workspace"] = "Possessions"
                st.session_state["game_choice"] = target["game_id"]
                st.session_state[f"possession_choice_{target['game_id']}"] = target["id"]
            st.button("Edit linked possession boundaries", key="edit_review_target", on_click=edit_target)
            st.caption("After editing, return to this batch, reload the saved review and explicitly reconfirm the changed link.")
        candidate_sets = st.session_state.setdefault("review_candidate_snapshots", {})
        candidates = candidate_sets.setdefault(prefix, {p["id"]: p for p in
                                                store.list("possession", game_id=frame['game_id'])})
        with st.form(prefix):
            notes = st.text_area("Correction / uncertainty notes", value=decision.get("notes", ""), key=f"{prefix}_notes")
            uncertain = st.checkbox("Uncertain case remains unresolved", value=decision.get("uncertain", False), key=f"{prefix}_uncertain")
            rebound = st.checkbox("Offensive rebound retained in this possession", value=decision.get("offensive_rebound", False), key=f"{prefix}_rebound")
            start = st.number_input("Corrected possession start (source seconds)", min_value=0.0,
                                    value=float(decision.get("draft_start", target['start'] if target else item['start'])), key=f"{prefix}_start")
            end = st.number_input("Corrected possession end (source seconds)", min_value=0.0,
                                  value=float(decision.get("draft_end", target['end'] if target else item['end'])), key=f"{prefix}_end")
            split = st.selectbox("Confirmed partition", ["train", "val"], index=int(decision.get("draft_split", item["split"]) == "val"), key=f"{prefix}_split")
            preferred_target = decision.get("draft_target_id") or (target['id'] if target else None)
            target_id = st.selectbox("Corrected target possession", ["new", *candidates],
                                     index=1 + list(candidates).index(preferred_target) if preferred_target in candidates else 0,
                                     format_func=lambda key: "Create separate manual possession" if key == "new" else
                                     f"{key} · {clock(candidates[key]['start'])}–{clock(candidates[key]['end'])}", key=f"{prefix}_target")
            confirmed = st.checkbox("I reviewed the whole possession boundaries and confirmed this partition", value=False,
                                    key=f"{prefix}_confirmed")
            save_notes = st.form_submit_button("Save review notes", key="save_review_notes")
            link = st.form_submit_button("Confirm corrected possession and link", key="confirm_review_possession", disabled=not available)
        if save_notes or link:
            try:
                if link:
                    if not confirmed:
                        raise ValueError("Explicitly review and confirm the whole possession boundaries and partition first.")
                    selected = candidates.get(target_id)
                    result = confirm_possession(store, batch, window_id, start=start, end=end, split=split,
                                                expected_revision=decision['revision'], frame_id=current, target_id=selected['id'] if selected else None,
                                                target_revision=selected['revision'] if selected else None,
                                                notes=notes, uncertain=uncertain, offensive_rebound=rebound)
                else:
                    selected = candidates.get(target_id)
                    result = save_frame_review(store, batch, current, notes=notes, uncertain=uncertain,
                        offensive_rebound=rebound, expected_revision=decision['revision'],
                        start=start, end=end, split=split, target_id=selected['id'] if selected else None,
                        target_revision=selected['revision'] if selected else None)
                decisions[f"{snapshot_key}:{current}"] = result
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        try:
            selection = frame_selection(store, batch, current)
        except ValueError as exc:
            st.info(f"Frame labels unavailable: {exc}")
        else:
            def open_frame():
                st.session_state["review_frame_selection"] = {"path": active, "frame_id": current}
                st.session_state["workspace"] = "Frame labels"
                st.session_state["game_choice"] = selection["game_id"]
                st.session_state[f"possession_choice_{selection['game_id']}"] = selection["possession_id"]
                possession = store.get("possession", selection["possession_id"])
                st.session_state[f"frame_time_{possession['id']}_{possession['revision']}"] = selection["time"]
            st.button("Open exact frame in Frame labels", key="open_review_frame", on_click=open_frame)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot open this review batch: {exc}")
