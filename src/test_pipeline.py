from datetime import datetime, timedelta
from src.data.clip_builder import build_clips
from src.data.shard_writer import write_shards


# ----------------------------
# Step 1 — Create Fake Operations
# ----------------------------

base_time = datetime.now()

operations = [
    {
        "operation": "Box Setup",
        "start": base_time,
        "end": base_time + timedelta(seconds=5)
    },
    {
        "operation": "Tape",
        "start": base_time + timedelta(seconds=5),
        "end": base_time + timedelta(seconds=10)
    },
    {
        "operation": "Put Items",
        "start": base_time + timedelta(seconds=10),
        "end": base_time + timedelta(seconds=15)
    }
]

# ----------------------------
# Step 2 — Build Clips
# ----------------------------

clips = build_clips(
    operations=operations,
    subject_id="U_TEST",
    session_id="S_TEST"
)

print("Generated clips:", len(clips))
print("First clip:", clips[0])


# ----------------------------
# Step 3 — Attach Fake Image Paths
# ----------------------------

for clip in clips:
    clip["image_paths"] = [
        f"U_TEST/S_TEST/frame_{frame:06d}.jpg"
        for frame in clip["sample_frames"]
    ]


# ----------------------------
# Step 4 — Write WebDataset Shards
# ----------------------------

write_shards(clips, shard_dir="data/shards_test", shard_size=2)

print("Pipeline test complete.")