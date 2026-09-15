import json
import sys
import io
import base64
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = Path('channel_intelligence_pipeline.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

# Adjust cell 4 (SCENARIO_PATH definition) and models path
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "SCENARIO_PATH = Path('../deepmimo_scenarios/asu_campus_3p5')" in src:
            src = src.replace(
                "SCENARIO_PATH = Path('../deepmimo_scenarios/asu_campus_3p5')",
                "candidates = [Path('deepmimo_scenarios/asu_campus_3p5'), Path('../deepmimo_scenarios/asu_campus_3p5')]\nSCENARIO_PATH = next((c for c in candidates if c.exists()), Path('deepmimo_scenarios/asu_campus_3p5'))"
            )
            cell['source'] = src.splitlines(keepends=True)
        if "../models" in src:
            src = src.replace("../models", "models")
            cell['source'] = src.splitlines(keepends=True)
        if "../eda_overview.png" in src:
            src = src.replace("../eda_overview.png", "eda_overview.png")
            cell['source'] = src.splitlines(keepends=True)
        if "../beam_results.png" in src:
            src = src.replace("../beam_results.png", "beam_results.png")
            cell['source'] = src.splitlines(keepends=True)
        if "../rss_results.png" in src:
            src = src.replace("../rss_results.png", "rss_results.png")
            cell['source'] = src.splitlines(keepends=True)
        if "../los_results.png" in src:
            src = src.replace("../los_results.png", "los_results.png")
            cell['source'] = src.splitlines(keepends=True)
        if "../ae_results.png" in src:
            src = src.replace("../ae_results.png", "ae_results.png")
            cell['source'] = src.splitlines(keepends=True)
        if "../feature_importance.png" in src:
            src = src.replace("../feature_importance.png", "feature_importance.png")
            cell['source'] = src.splitlines(keepends=True)

# Now execute all code cells sequentially in a shared global namespace and record outputs
glob_ns = {}
exec_count = 1

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    code = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    print(f"Executing code cell {idx+1}/{len(nb['cells'])}...")
    
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    
    outputs = []
    try:
        # Clear any existing plots
        plt.close('all')
        
        # Execute code in persistent namespace
        exec(code, glob_ns)
        
        # Check if matplotlib generated any open figures
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
            img_buf.seek(0)
            b64_img = base64.b64encode(img_buf.read()).decode('utf-8')
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64_img,
                    "text/plain": "<Figure size ... with ... Axes>"
                },
                "metadata": {}
            })
            plt.close(fig)
            
        std_text = stdout_buf.getvalue()
        if std_text:
            outputs.insert(0, {
                "output_type": "stream",
                "name": "stdout",
                "text": std_text.splitlines(keepends=True)
            })
            
    except Exception as e:
        std_text = stdout_buf.getvalue()
        if std_text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": std_text.splitlines(keepends=True)
            })
        outputs.append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": [f"{type(e).__name__}: {str(e)}"]
        })
        print(f"  [ERROR] in cell {idx+1}: {e}")
    finally:
        sys.stdout = old_stdout
        
    cell['outputs'] = outputs
    cell['execution_count'] = exec_count
    exec_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nSuccessfully executed and updated channel_intelligence_pipeline.ipynb with all outputs!")
