"""Generate all data-driven charts from CSV experiment results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = r"D:\courses\CS181\final_proj\2048_cs181\final_results\pictures"
plt.rcParams['font.size'] = 10

# ====== [Fig 1] Expectimax Ablation Bar Chart ======
def fig1_expectimax():
    configs = ['1-A\nGreedy', '1-B\nStd+Heur', '1-C\nAS+Heur',
               '1-D\nStd+StateNT', '1-E\nStd+AfterNT\n(mismatch!)', '1-F\nAS+StateNT',
               '1-G\nAS+AfterNT', '1-H\nAS+ANT\n+BeamSearch']
    scores = [4779, 11059, 11952, 64740, 5066, 60126, 114885, 125721]
    times  = [0.11, 4.93, 0.64, 26.81, 34.47, 2.56, 2.59, 2.31]
    colors = ['#cccccc', '#cccccc', '#cccccc',
              '#7a8faa', '#b88080', '#7a8faa',
              '#4a8a4a', '#3a7a3a']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    bars = ax1.bar(range(len(configs)), scores, color=colors, edgecolor='white', linewidth=0.8)
    ax1.set_xticks(range(len(configs))); ax1.set_xticklabels(configs, fontsize=8.5)
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_title('Expectimax 2x2 Ablation Matrix: Score (Depth=2, 100 games)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, s in zip(bars, scores):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1500,
                 f'{s:,}', ha='center', fontsize=7.5, fontweight='bold')

    bars2 = ax2.bar(range(len(configs)), times, color=colors, edgecolor='white', linewidth=0.8)
    ax2.set_xticks(range(len(configs))); ax2.set_xticklabels(configs, fontsize=8.5)
    ax2.set_ylabel('Time per Move (ms)', fontsize=12)
    ax2.set_title('Expectimax 2x2 Ablation Matrix: Time per Move', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{t:.2f}', ha='center', fontsize=7.5, fontweight='bold')

    ax1.annotate('Mismatch catastrophe\n(-92% vs 1-G)', xy=(4, 5066), xytext=(2.3, 30000),
                arrowprops=dict(arrowstyle='->', color='#b88080'), fontsize=9, color='#b88080', fontweight='bold')
    ax1.annotate('Full decoupling\n(+77% vs 1-D)', xy=(6, 114885), xytext=(5, 140000),
                arrowprops=dict(arrowstyle='->', color='#2a6a2a'), fontsize=9, color='#2a6a2a', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart_fig1_expectimax.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("OK fig1_expectimax")

# ====== [Fig 3] MCTS Score vs Simulation Budget ======
def fig3_mcts():
    sims = [200, 500, 1000, 2000]
    state = [48548, 58056, 48642, 58614]
    after = [45414, 54280, 60497, 65934]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sims, state, 'o-', color='#7a8faa', linewidth=2.5, markersize=10, label='State MCTS')
    ax.plot(sims, after, 's-', color='#4a8a4a', linewidth=2.5, markersize=10, label='Afterstate MCTS')
    ax.axvspan(200, 750, alpha=0.06, color='#b88080')
    ax.axvspan(750, 2000, alpha=0.06, color='#4a8a4a')

    for i in range(len(sims)):
        ax.annotate(f'{state[i]:,}', (sims[i], state[i]),
                    textcoords="offset points", xytext=(15, -18), fontsize=9, color='#7a8faa')
        ax.annotate(f'{after[i]:,}', (sims[i], after[i]),
                    textcoords="offset points", xytext=(15, 8), fontsize=9, color='#2a6a2a')

    # Annotations
    ax.annotate('Afterstate LOSES\nat low budgets', xy=(350, 50500), fontsize=10,
                color='#b88080', fontweight='bold', ha='center')
    ax.annotate('Afterstate WINS\nat high budgets', xy=(1400, 50500), fontsize=10,
                color='#2a6a2a', fontweight='bold', ha='center')

    # Note non-monotonic drop
    ax.annotate('Non-monotonic\ndrop (noise!)', xy=(1000, 48642), xytext=(850, 43000),
                arrowprops=dict(arrowstyle='->', color='#7a8faa'), fontsize=8.5, color='#7a8faa')

    ax.set_xlabel('Simulation Budget (rollouts per move)', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('MCTS: State vs Afterstate by Simulation Budget (Heuristic evaluator)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(sims)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart_fig3_mcts.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("OK fig3_mcts")

# ====== [Fig 8] Phase 3 TD Comparison ======
def fig8_phase3_td():
    configs = ['3-D\nV+Sampling\n(Baseline)', '3-E\nMV+Sampling',
               '3-F\nTDA-Full\n(Full Expectation)', '3-G\nDownside-MV\n(Downside Only)']
    scores = [18731, 18768, 17767, 17937]
    bias = [1.976, 1.934, 2.355, 1.950]
    colors = ['#4a8a4a', '#b8a878', '#b88080', '#b88080']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars = ax1.bar(range(len(configs)), scores, color=colors, edgecolor='white', linewidth=0.8)
    ax1.set_xticks(range(len(configs))); ax1.set_xticklabels(configs, fontsize=8.5)
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_title('Phase 3 TD: Score (100K episodes)', fontsize=13, fontweight='bold')
    ax1.axhline(y=18731, color='#4a8a4a', linestyle='--', alpha=0.4, linewidth=1.5, label='3-D baseline')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, s in zip(bars, scores):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                 f'{s:,}', ha='center', fontsize=9, fontweight='bold')

    bars2 = ax2.bar(range(len(configs)), bias, color=colors, edgecolor='white', linewidth=0.8)
    ax2.set_xticks(range(len(configs))); ax2.set_xticklabels(configs, fontsize=8.5)
    ax2.set_ylabel('Norm Bias (RoM)', fontsize=12)
    ax2.set_title('Phase 3 TD: Overestimation Bias', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=1.976, color='#4a8a4a', linestyle='--', alpha=0.4, linewidth=1.5, label='3-D baseline')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, b in zip(bars2, bias):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                 f'+{b:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart_fig8_phase3td.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("OK fig8_phase3td")

# ====== [Fig 9] MCTS Topology ======
def fig9_mcts_topology():
    configs = ['Heuristic\nState', 'Heuristic\nAfterstate',
               'N-Tuple\nState', 'N-Tuple\nAfterstate']
    scores = [29276, 33272, 90636, 135712]
    macro_depth = [5.22, 5.83, 5.20, 5.71]
    probe_entropy = [0.221, 0.235, 0.330, 0.318]
    colors = ['#b88080', '#b88080', '#7a8faa', '#4a8a4a']
    markers = ['o', 's', 'o', 's']
    labels = ['Heuristic State', 'Heuristic After', 'N-Tuple State', 'N-Tuple After']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(len(configs)):
        size = scores[i] / 400
        ax1.scatter(macro_depth[i], probe_entropy[i], s=size, c=colors[i],
                    marker=markers[i], label=labels[i], edgecolors='white', linewidth=1.5, zorder=5)
        ax1.annotate(f'{scores[i]:,}', (macro_depth[i], probe_entropy[i]),
                     textcoords="offset points", xytext=(0, 15), fontsize=8,
                     color=colors[i], fontweight='bold', ha='center')

    ax1.annotate('', xy=(5.71, 0.318), xytext=(5.20, 0.330),
                arrowprops=dict(arrowstyle='->', color='#4a8a4a', lw=2))
    ax1.annotate('Deeper +\nMore focused', xy=(5.5, 0.324), fontsize=8, color='#2a6a2a', fontweight='bold')
    ax1.set_xlabel('Macro Depth (higher = deeper search)', fontsize=12)
    ax1.set_ylabel('Probe Entropy (lower = more focused)', fontsize=12)
    ax1.set_title('MCTS Search Topology (bubble size = score)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.invert_yaxis()

    bars = ax2.bar(range(len(configs)), scores, color=colors, edgecolor='white', linewidth=0.8)
    ax2.set_xticks(range(len(configs))); ax2.set_xticklabels(configs, fontsize=10)
    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_title('MCTS + N-Tuple: Performance Leap (2000 rollouts)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, s in zip(bars, scores):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000,
                 f'{s:,}', ha='center', fontsize=10, fontweight='bold')
    ax2.axhline(y=114885, color='#7a8faa', linestyle='--', alpha=0.4, linewidth=1.5)
    ax2.text(3.5, 117000, 'Expectimax\n115K', fontsize=8, color='#7a8faa', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart_fig9_topology.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("OK fig9_topology")

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    fig1_expectimax()
    fig3_mcts()
    fig8_phase3_td()
    fig9_mcts_topology()
    print("\nAll 4 data charts generated successfully!")
