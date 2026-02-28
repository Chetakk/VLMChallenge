from datetime import datetime, timedelta
from src.data.clip_builder import build_clips
from src.data.shard_writer import write_shards



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


clips = build_clips(
    operations=operations,
    subject_id="U_TEST",
    session_id="S_TEST"
)

print("Generated clips:", len(clips))
print("First clip:", clips[0])



for clip in clips:
    clip["image_paths"] = [
        f"U_TEST/S_TEST/frame_{frame:06d}.jpg"
        for frame in clip["sample_frames"]
    ]


write_shards(clips, shard_dir="data/shards_test", shard_size=2)
print("Pipeline test complete.")