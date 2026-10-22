# Faster frame review

In **Review batches**, link the sampled frame to its checked whole possession,
then choose **Open exact frame in Frame labels**. Several samples may share one
possession. Sampling windows are not possessions; keep offensive rebounds together.

The default **Click on frame** mode needs two opposite corners around the object.
Select **ball** or **rim**, click the two corners, inspect the cyan draft, then
choose **Save reviewed box**. Escape cancels an unfinished first corner. Drawing
does not write a label. **Fine-tune coordinates** remains available when needed.
Use **Box to edit** to select an existing saved box before changing it.

At an exact batch timestamp with existing model suggestions, orange numbered boxes
have three actions:

- **Accept box** saves that box as a human-verified annotation.
- **Reject box** saves a separate rejection decision, not a negative-frame label.
- **Correct box** copies it into a draft. Saving the correction resolves the
  suggestion atomically, preventing duplicate acceptance of the original box.

Some frames have no proposals. Add missing objects with the click tool. The
existing COCO proposals cover balls, not rims. The new public data package is not
a trained model and does not automatically generate new suggestions.

Review **every visible ball and rim** in a chosen still. Only then choose **Mark
complete frame reviewed**, or use **Reviewed negative** after checking there are
no relevant objects. Unsaved draft changes must be saved or discarded first.
Uncertain work can remain unreviewed. Reload the frame editor after a concurrent
change; reloading refreshes coordinates and revision checks together.

The batch has 50 review windows and 300 sampled stills, not 300 video clips or
every video frame. Start with five windows / 30 stills; existing custom-time
annotations remain useful additional examples. Do not move their timestamps.

Local software tests exercise coordinate conversion and acceptance persistence.
Actual browser interaction still requires a manual check while local browser
control is disabled. No navigation, drawing, acceptance or rejection starts
training or exports a dataset.
