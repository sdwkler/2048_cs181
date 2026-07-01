"""Extract Mermaid diagrams from HTML and render to PNG using mmdc."""
import re, subprocess, os

html_path = "../浅层架构图.html"
out_dir = "pictures"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Find all mermaid blocks
blocks = re.findall(r'<div class="mermaid">\s*(.*?)\s*</div>', content, re.DOTALL)

# Map block indices to descriptive names
names = [
    "framework_3phase",      # 0: 三阶段递进
    "mdp_traditional",       # 1: 传统 MDP
    "mdp_afterstate",        # 2: Afterstate MDP
    "ablation_2x2_matrix",   # 3: 2x2 消融矩阵
    "theory_method_map",     # 4: 理论→方法映射
    "platform_5layer",       # 5: 五层架构
    "precision_gradient",    # 6: 精度梯度
]

for i, (block, name) in enumerate(zip(blocks, names)):
    mmd_path = os.path.join(out_dir, f"{name}.mmd")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(block)

    png_path = os.path.join(out_dir, f"{name}.png")
    print(f"Rendering {name}...")
    result = subprocess.run([
        "mmdc", "-i", mmd_path, "-o", png_path,
        "-w", "1200", "-b", "white",
        "--theme", "neutral",
        "--pdfFit"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
    else:
        print(f"  OK -> {png_path}")

print("Done!")
