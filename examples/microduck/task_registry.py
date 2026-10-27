"""Names for exactly the variants registered by the bundled official project."""

BASE_TASKS = (
    "velocity",
    "velstand",
    "standup",
    "sitstand",
    "groundpick",
    "ballkick",
    "rollers",
    "swizzle",
    "rollercrouch",
    "rollerslope",
    "rollerstandup",
    "spin",
    "roulade",
)
ROUGH_TASKS = ("velocity", "velstand", "standup", "sitstand", "groundpick")
BACKLASH_TASKS = BASE_TASKS[:6] + ("rollers", "swizzle", "rollercrouch", "rollerslope")
TASKS = (
    BASE_TASKS
    + tuple(f"{t}_rough" for t in ROUGH_TASKS)
    + tuple(f"{t}_backlash" for t in BACKLASH_TASKS)
    + tuple(f"{t}_rough_backlash" for t in ROUGH_TASKS)
)
ROLLER_TASKS = ("rollers", "swizzle", "rollercrouch", "rollerslope", "rollerstandup", "spin")


def base_task(task):
    return task.split("_")[0]


def task_key(official_id):
    if official_id.endswith("-Rollers"):
        name = "rollers"
    elif "-Swizzle-" in official_id:
        name = "swizzle"
    else:
        name = official_id.split("-")[1].lower()
    return name + ("_rough" if "-Rough-" in official_id else "") + ("_backlash" if "-Backlash-" in official_id else "")
