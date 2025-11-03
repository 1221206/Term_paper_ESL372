import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    """Creates and saves a diagram of the project's architecture using Matplotlib."""
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Define styles
    box_style = dict(boxstyle='round,pad=0.5', fc='lightblue', ec='black', lw=1.5)
    arrow_style = dict(arrowstyle='->', ec='black', lw=1.5)
    data_style = dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='black', lw=1.5)
    group_style = dict(fill=False, ec='black', lw=2)

    # Helper function for creating boxes and arrows
    def draw_box(text, xy, style=box_style):
        return ax.text(xy[0], xy[1], text, ha='center', va='center', bbox=style, fontsize=9)

    def draw_arrow(box1, box2, rad=0.1):
        # Get the center positions of the boxes
        pos1 = (box1.get_position()[0], box1.get_position()[1])
        pos2 = (box2.get_position()[0], box2.get_position()[1])
        
        # Use a simple arrow for now, as precise bbox calculation is complex
        ax.add_patch(patches.FancyArrowPatch(pos1, pos2, **arrow_style, shrinkA=8, shrinkB=8, connectionstyle=f'arc3,rad={rad}'))

    # --- Define Box Positions ---
    pos = {
        'raw_data': (20, 90),
        'loader': (20, 80),
        'norm_data': (20, 70),
        'sol_u': (50, 90),
        'pred_u': (50, 80),
        'ad': (40, 70),
        'grads': (40, 60),
        'dyn_f': (60, 70),
        'resid': (60, 60),
        'loss1': (80, 90),
        'loss2': (80, 80),
        'loss3': (80, 70),
        'total_loss': (80, 55),
        'optimizer': (80, 45),
        'script': (20, 25),
        'model_out': (50, 25),
        'preds_out': (50, 15),
        'final_out': (80, 15)
    }

    # --- Draw Boxes ---
    boxes = {
        'raw_data': draw_box('Raw Data\n(.csv files)', pos['raw_data']),
        'loader': draw_box('dataloader.py', pos['loader']),
        'norm_data': draw_box('Paired & Normalized Data', pos['norm_data'], style=data_style),
        'sol_u': draw_box('solution_u Network', pos['sol_u']),
        'pred_u': draw_box('Predicted SOH (u)', pos['pred_u'], style=data_style),
        'ad': draw_box('Automatic Differentiation', pos['ad']),
        'grads': draw_box('Derivatives (u_t, u_x)', pos['grads'], style=data_style),
        'dyn_f': draw_box('dynamical_F Network', pos['dyn_f']),
        'resid': draw_box('Physics Residual (F)', pos['resid'], style=data_style),
        'loss1': draw_box('loss1: Data Fidelity', pos['loss1']),
        'loss2': draw_box('loss2: Physics Residual', pos['loss2']),
        'loss3': draw_box('loss3: Monotonicity', pos['loss3']),
        'total_loss': draw_box('Total Loss', pos['total_loss'], style=dict(boxstyle='circle', fc='lightpink', ec='black')),
        'optimizer': draw_box('Optimizer', pos['optimizer']),
        'script': draw_box('final_KAN_PINN_plots.py', pos['script']),
        'model_out': draw_box('Final Trained Model', pos['model_out']),
        'preds_out': draw_box('Final Predictions', pos['preds_out'], style=data_style),
        'final_out': draw_box('Final Output\n(Plot & Metrics)', pos['final_out'], style=dict(boxstyle='round,pad=0.5', fc='palegreen', ec='black'))
    }

    # --- Draw Arrows ---
    draw_arrow(boxes['raw_data'], boxes['loader'])
    draw_arrow(boxes['loader'], boxes['norm_data'])
    draw_arrow(boxes['norm_data'], boxes['sol_u'])
    draw_arrow(boxes['sol_u'], boxes['pred_u'])
    draw_arrow(boxes['pred_u'], boxes['ad'])
    draw_arrow(boxes['ad'], boxes['grads'])
    draw_arrow(boxes['grads'], boxes['dyn_f'])
    draw_arrow(boxes['pred_u'], boxes['dyn_f'])
    draw_arrow(boxes['dyn_f'], boxes['resid'])
    draw_arrow(boxes['pred_u'], boxes['loss1'])
    draw_arrow(boxes['resid'], boxes['loss2'])
    draw_arrow(boxes['pred_u'], boxes['loss3'])
    draw_arrow(boxes['loss1'], boxes['total_loss'])
    draw_arrow(boxes['loss2'], boxes['total_loss'])
    draw_arrow(boxes['loss3'], boxes['total_loss'])
    draw_arrow(boxes['total_loss'], boxes['optimizer'])
    draw_arrow(boxes['optimizer'], boxes['sol_u'], rad=-0.2)
    draw_arrow(boxes['optimizer'], boxes['dyn_f'], rad=0.2)
    draw_arrow(boxes['script'], boxes['model_out'])
    draw_arrow(boxes['model_out'], boxes['preds_out'])
    draw_arrow(boxes['preds_out'], boxes['final_out'])

    # --- Draw Group Boxes ---
    ax.add_patch(patches.Rectangle((5, 60), 30, 35, **group_style, edgecolor='dodgerblue'))
    ax.text(20, 97, '1. Data Layer', ha='center', va='center', fontsize=12)
    ax.add_patch(patches.Rectangle((35, 50), 30, 45, **group_style, edgecolor='grey'))
    ax.text(50, 97, '2. Model Layer', ha='center', va='center', fontsize=12)
    ax.add_patch(patches.Rectangle((70, 40), 20, 55, **group_style, edgecolor='seagreen'))
    ax.text(80, 97, '3. Training & Loss', ha='center', va='center', fontsize=12)
    ax.add_patch(patches.Rectangle((5, 5), 90, 30, **group_style, edgecolor='goldenrod'))
    ax.text(50, 37, '4. Execution & Output', ha='center', va='center', fontsize=12)

    plt.title('Project Architecture Flow', fontsize=20)
    plt.savefig('final_architecture.png', dpi=300, bbox_inches='tight')
    print("Successfully created 'final_architecture.png' using Matplotlib.")

if __name__ == '__main__':
    create_architecture_diagram()