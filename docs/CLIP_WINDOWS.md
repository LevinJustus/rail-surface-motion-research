# Video clip windows with explicit resets

The input CSV has four columns: asset ID, verified source group ID, split, and decoded frame count. The example below uses invented metadata.

~~~csv
asset_id,group_id,split,frame_count
example_a,rail_a,train,8
example_b,rail_b,train,5
example_c,rail_c,test,10
~~~

Save the example as assets.csv and run:

~~~text
python tools/real_video/build_clip_windows.py assets.csv example_output --split train --seed 7 --frames-per-window 3
~~~

The clip order is reproducible. The playback mapping uses inclusive zero-based offsets. Every model window remains within one source clip and declares a reset. Share a group ID across repeated observations of the same rail section or acquisition group. Assigned examples need a group and a group cannot cross partitions. The unassigned partition supports preview order and produces no training windows.

The tool handles indexing. Review and confirm candidate masks before using the indexed observations as segmentation labels. Run the model on the original frames. A re-encoded montage is intended for review.

## Review screenshot positions in an original clip

The MATLAB function match_source_frames.m uses VideoReader, Image Processing Toolbox and Computer Vision Toolbox on 1920 × 1080 video. In a new local run directory, prepare intake.csv, and save each supplied screenshot as attachments/ followed by the reference ID and .png. Store the corresponding originals under the archive's originals directory.

~~~csv
reference,source_video,user_time_s,attachment
Example01,example.mp4,2,example.png
~~~

Call match_source_frames(runDir, archiveDir) for candidate frames, then match_source_frames(runDir, archiveDir, true) to refine the candidates against the original frames. The frame index and screenshot-region coordinates are zero-based. The boxes localize the screenshot region for review. Confirm the actual defect region and any repeated physical event before using labels for evaluation or training.
