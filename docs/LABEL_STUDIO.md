# Label your own frames in Label Studio

The public datasets taught the model what a basketball looks like in general, but on
this gym's footage it still calls shoes and floor spots "ball". A few hundred labeled
frames from your own game, including frames with nothing in them, fix that.

## 1. Pull frames (already done once)

```sh
uv run --extra vision python scripts/label_studio_frames.py export
```

This writes `artifacts/labeling/my-frames-v1/` with `images/`, `tasks.json`,
`label_config.xml` and `manifest.json`. It uses only the game's declared train and val
blocks, staying 10 s away from block edges, and never reads the locked test block.
About half the frames have a ball near a rim, which is where the model struggles
most. The rest are spread over live play. Every frame comes with the current model's
boxes as suggestions to correct.

## 2. Start the two local servers

```sh
uv run python scripts/label_studio_frames.py serve
```

```sh
label-studio start --port 8080
```

(`label-studio` was installed with `uv tool install label-studio`.)

## 3. Create the project (one time)

1. Open http://localhost:8080 and sign up. This is a local account on your computer.
2. **Create Project**, name it `my-frames-v1`.
3. **Labeling Setup** → **Custom template** → **Code**, then paste the contents of
   `artifacts/labeling/my-frames-v1/label_config.xml` and save.
4. **Data Import** → upload `artifacts/labeling/my-frames-v1/tasks.json`.

## 4. Label

- Open a frame. The model's suggestions are already drawn.
- **Delete** wrong boxes (shoes, floor spots, heads), **fix** loose ones, and **add**
  any missed ball or rim: press `1` for ball or `2` for rim, then drag.
- Box every visible ball and every visible rim, including blurry or partly hidden
  ones. Skip the scoreboard.
- **Submit even when the frame is empty.** An empty, submitted frame teaches the
  model that nothing there is a ball. Use **Skip** only when you can't tell.
- Roughly 5–10 seconds per frame, so 375 frames take about an hour. Exporting
  partway through is fine; unlabeled frames are left out.

## 5. Export and convert

In the project: **Export** → **JSON** → save the file, then:

```sh
uv run python scripts/label_studio_frames.py convert path/to/export.json
```

This writes `artifacts/datasets/my-frames-v1/` and `artifacts/datasets/my-frames-v1.zip`.

## 6. Train

Upload `my-frames-v1.zip` to `MyDrive/PlayLab/` and run the Colab cell
(`notebooks/train_ball_rim_cell.py`). It is already set to run `ball-rim-yolo11s-02`.
That run continues from the current model, combines the public data with your frames
(weighted 3× in training), and reports `own_val_*` scores on your val-block frames.
Those are the numbers that reflect your footage.
