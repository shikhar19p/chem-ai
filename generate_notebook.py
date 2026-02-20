import json, os

base = r"C:\Users\shikhar pulluri\Desktop\murali projject"
out_path = os.path.join(base, "notebooks", "reactive_extraction_pipeline.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

cells = []
cc = [0]

def md(src):
    cid = "md" + str(cc[0]); cc[0] += 1
    return {"cell_type": "markdown", "metadata": {}, "source": src, "id": cid}

def code(src):
    cid = "c" + str(cc[0]); cc[0] += 1
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src, "id": cid}
