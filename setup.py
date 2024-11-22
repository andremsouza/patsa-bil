import os

with open(".env", "w") as f:
    f.write(f"PACKAGEPATH={os.getcwd()}\n")
