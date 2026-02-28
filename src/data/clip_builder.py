from typing import List, Dict
import numpy as np

FPS = 25
CLIP_Duration = 5
CLIP_FRAME_Count = FPS * CLIP_Duration
HALF_CLIP = CLIP_FRAME_Count // 2
NUM_Sample_Frames = 8



def build_clips(operations: List[Dict], subject_id: str, session_id: str) -> List[Dict]:


    if len(operations) < 2:
        return []
    
    #convert timestamps to frames

    session_start_time = operations[0]["start"]

    for op in operations:
        start_seconds = (op["start"] - session_start_time).total_seconds()
        end_seconds = (op["end"] - session_start_time).total_seconds()

        op["start_frame"] = int(start_seconds * FPS)
        op["end_frame"] = int(end_seconds * FPS)

    clips = []


    #build seq anticipation samples    

    for i in range(len(operations) - 1):
        current_op = operations[i]
        next_op = operations[i+1]

        boundary_frame = current_op["end_frame"]

        clip_start = max(0, boundary_frame - HALF_CLIP)
        clip_end = boundary_frame + HALF_CLIP

        sample_frames = np.linspace(clip_start, clip_end, NUM_Sample_Frames).astype(int).tolist()


        clip = {
            "clip_id": f"{subject_id}_{session_id}_{i:04d}",
            "dominant_operation": current_op["operation"],
            "tempopral_segment" : {
                "start_frame": current_op["start_frame"],
                "end_frame": current_op["end_frame"],
            },
            "anticipated_next_operation": next_op["operation"],
            "sample_frames": sample_frames
        }


        clips.append(clip)

    return clips



#for testing


# from annotation_parser import load_operations
# # from clip_builder import build_clips

# ops = load_operations("C:/Users/Chetak/Documents/GitHub/projects/VLM Challenge/Dataset/U0101/annotation/openpack-operations/S0100.csv")

# clips = build_clips(ops, subject_id="U0101", session_id="S0100")

# print("Total clips:", len(clips))
# print(clips[0])

