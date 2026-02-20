import base64, os
p = r"C:\Users\shikhar pulluri\Desktop\murali projject\gen_nb_test.py"
existing = open(p, "rb").read()
b64 = open(r"C:\Users\shikhar pulluri\Desktop\murali projject\nb_append.b64","r").read().strip()
import base64
append_bytes = base64.b64decode(b64)
open(p,"wb").write(existing + append_bytes)
print("Script updated, size:", os.path.getsize(p))
