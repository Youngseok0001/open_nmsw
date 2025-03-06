"""
Set some commonly used strings as objects. this prevents me from making spelling mistakes, e.g) validation, val, valid
"""

TRAIN = "train"
VALID = "valid"
TEST = "test"

LOCAL = "local"  # full-res vol
GLOBAL = "global"  # low-res vol
PATCH = "patch"  # patches from local

LOGIT = "logit"
LOSS = "loss"
LOCAL_LOSS = "local_loss"
GLOBAL_LOSS = "global_loss"
TOTAL_LOSS = "total_loss"

VOL = "vol"
LAB = "lab"
CASE_ID = "case_id"
PRED = "pred"

LOCAL_VOL = LOCAL + "_" + VOL
LOCAL_LAB = LOCAL + "_" + LAB
LOCAL_LOGIT = LOCAL + "_" + LOGIT
LOCAL_PRED = LOCAL + "_" + PRED
LOCAL_LOSS = LOCAL + "_" + LOSS

GLOBAL_VOL = GLOBAL + "_" + VOL
GLOBAL_LAB = GLOBAL + "_" + LAB
GLOBAL_LOGIT = GLOBAL + "_" + LOGIT
GLOBAL_PRED = GLOBAL + "_" + PRED
GLOBAL_LOSS = GLOBAL + "_" + LOSS

PATCH_VOL = PATCH + "_" + VOL
PATCH_LAB = PATCH + "_" + LAB
PATCH_LOGIT = PATCH + "_" + LOGIT
PATCH_PRED = PATCH + "_" + PRED
PATCH_LOSS = PATCH + "_" + LOSS
