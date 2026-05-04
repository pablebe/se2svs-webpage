import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# rank_colors = {
#     1: '#66bd63',  # Dark Green
#     2: '#a6d96a',  # Medium Green
#     3: '#fee08b',  # Light orange yellow 
#     4: "#e8b070",  # Light orange
#     5: "#dc5d5d",  # Light red
# }

# # Load CSV
# df = pd.read_csv("metric_dmos_companion_table_all_files.csv", sep=';')

# # Target models and metrics
# models = [
#     'htdemucs',
#     'melroformer_small',
#     'melroformer_large',
#     'sgmsvs',
#     'melroformer_bigvgan'
# ]

# model_display_dict = {
#     'htdemucs': 'HTDemucs',
#     'melroformer_small': 'Mel-RoFo. (S)',
#     'melroformer_large': 'Mel-RoFo. (L)',
#     'sgmsvs': 'SGMSVS',
#     'melroformer_bigvgan': 'Mel-RoFo. (S) + BigVGAN'
# }

# metrics = ['sdr', 'emb_mse_mert_df', 'DMOS']
# metric_display_dict = {'sdr': 'SDR', 'emb_mse_mert_df': 'M-L12 MSE', 'DMOS': 'DMOS'}
# higher_is_better = {
#     'sdr': True,
#     'emb_mse_mert_df': False,
#     'DMOS': True
# }
# # Step 1: Group data into a dict: {file_id: {model: {metric: value}}}
# data_by_file = {}
# for _, row in df.iterrows():
#     fid = row['file_id']
#     model = row['model_name']
#     if fid not in data_by_file:
#         data_by_file[fid] = {}
#     data_by_file[fid][model] = {
#         metric: row[metric] for metric in metrics
#     }

# # Step 2: Build HTML
# html = ['<table border="1" cellspacing="0" cellpadding="4" style="border-collapse: collapse; text-align: center;">']

# # Header rows
# html.append('<tr><th rowspan="2">file_id</th>')
# for model in models:
#     html.append(f'<th colspan="3">{model_display_dict[model]}</th>')
# html.append('</tr>')

# html.append('<tr>')
# for _ in models:
#     for metric in metrics:
#         html.append(f'<th>{metric_display_dict[metric]}</th>')
# html.append('</tr>')

# round_dict = {'sdr':2, 'emb_mse_mert_df':4, 'DMOS':2}

# # Data rows with per-row ranking
# for fid in sorted(data_by_file.keys()):
#     model_data = data_by_file[fid]
    
#     # Prepare rank lookup per metric for this file_id
#     ranks_per_metric = {}
#     for metric in metrics:
#         values = []
#         for model in models:
#             val = model_data.get(model, {}).get(metric, None)
#             if val is not None:
#                 values.append((model, val))
#         # Sort and rank
#         values_sorted = sorted(values, key=lambda x: x[1], reverse=higher_is_better[metric])
#         ranks = {model: rank + 1 for rank, (model, _) in enumerate(values_sorted)}
#         ranks_per_metric[metric] = ranks

#     # Build row
#     fid_display = '#'+str(fid).split('_')[-1] if isinstance(fid, str) else fid
#     html.append(f'<tr><td>{fid}</td>')
#     for model in models:
#         for metric in metrics:
#             val = model_data.get(model, {}).get(metric, None)
#             val = round(val, round_dict[metric]) if isinstance(val, float) else val
#             rank = ranks_per_metric[metric].get(model, None)
#             color = f'background-color: {rank_colors[rank]};' if rank in rank_colors else ''
#             comma_num = round_dict[metric]
#             display_val = f"{val:.{comma_num}f}" if isinstance(val, float) else (str(val) if val is not None else '')
#             html.append(f'<td style="{color}">{display_val}</td>')
#     html.append('</tr>')

# html.append('</table>')

# # Write to file
# with open("per_row_ranked_table.html", "w") as f:
#     f.write('\n'.join(html))

# print("HTML table saved to 'per_row_ranked_table.html'")



#%% Table for whole data per discriminative and generative models
# Read CSV with ; delimiter


# Load your CSV file
df = pd.read_csv("/Users/Paul/IEM-Phd/01_PhD/06_Conferences/WASPAA2025/zenodo_upload/gensvs_eval_data.csv")
#df = df.drop(columns=['Unnamed: 0'])
df = df[['file_id', 'model_name', 'sdr', 'emb_mse_mert_df', 'DMOS', 'singmos', 'emb_mse_music2latent_df']]

# Define model groups
group_a = ['htdemucs', 'melroformer_small', 'melroformer_large']
group_b = ['melroformer_bigvgan', 'sgmsvs']

# Assign group labels
def assign_group(model):
    if model in group_a:
        return 'Discriminative'
    elif model in group_b:
        return 'Generative'
    else:
        return 'Unknown'

df['group'] = df['model_name'].apply(assign_group)

# Assign unique ID to each row for column headers
df['unique_id'] = ['row_' + str(i) for i in range(len(df))]

# Rank metrics within each group
df['SDR'] = df.groupby('group')['sdr'].rank(ascending=False, method='min')
df['MERT-L12 MSE'] = df.groupby('group')['emb_mse_mert_df'].rank(ascending=True, method='min')
df['DMOS'] = df.groupby('group')['DMOS'].rank(ascending=False, method='min')
df['SINGMOS'] = df.groupby('group')['singmos'].rank(ascending=False, method='min')
df['MUSIC2LATENT MSE'] = df.groupby('group')['emb_mse_music2latent_df'].rank(ascending=True, method='min')

# Keep only necessary columns
df_ranks = df[['unique_id', 'group', 'SINGMOS', 'SDR', 'MUSIC2LATENT MSE', 'MERT-L12 MSE', 'DMOS']]

# Create and sort rank tables
def create_sorted_rank_table(group_name):
    group_df = df_ranks[df_ranks['group'] == group_name]
    table = group_df.set_index('unique_id')[['SINGMOS', 'SDR', 'MUSIC2LATENT MSE', 'MERT-L12 MSE', 'DMOS']].T
    dmos_ranks = table.loc['DMOS']
    table = table[dmos_ranks.sort_values().index]
    return table

# Generate tables
table_a = create_sorted_rank_table('Discriminative')
table_b = create_sorted_rank_table('Generative')

def highlight_ranks_inline(df):
    cmap = plt.get_cmap("RdYlGn_r")  # green = good

    html = '<table>\n<tr><th></th>' + ''.join('<th></th>' for _ in df.columns) + '</tr>\n'

    for idx, row in df.iterrows():
        html += f'<tr><td style="text-align:right"><b>{idx}</b></td>'
        min_rank = row.min()
        max_rank = row.max()

        for i, val in enumerate(row):
            norm_val = (val - min_rank) / (max_rank - min_rank + 1e-9)
            rgb = cmap(norm_val)[:3]
            hex_color = '#%02x%02x%02x' % tuple(int(255 * c) for c in rgb)

            # Apply top/bottom borders from second column onward
            if i == 0:
                border_style = 'border-top: 2px solid black; border-bottom: 2px solid black; border-left: 1px solid black;'
                html += f'<td style="background-color: {hex_color}; {border_style} text-align: center;">&nbsp;</td>'        
            elif i == len(row) - 1:
                border_style = 'border-top: 2px solid black; border-bottom: 2px solid black; border-right: 1px solid black;'
                html += f'<td style="background-color: {hex_color}; {border_style} text-align: center;">&nbsp;</td>'        
            else:
                border_style = 'border-top: 2px solid black; border-bottom: 2px solid black;'
                html += f'<td style="background-color: {hex_color}; {border_style} text-align: center;">&nbsp;</td>'
        html += '</tr>\n'
    num_audiofiles = len(df.columns)-20
    html += '<tr><td></td><td colspan="10" style="text-align:left; border-left: 1px solid black">rank #1</td><td colspan='+str(num_audiofiles)+'></td><td colspan="10" style="text-align:right; border-right: 1px solid black">rank #'+str(len(df.columns))+'</td></tr>'

    html += '</table>\n'
    return html

# Generate HTML tables
html_a = highlight_ranks_inline(table_a)
html_b = highlight_ranks_inline(table_b)

# Save final HTML file
with open("rank_tables.html", "w", encoding="utf-8") as f:
    f.write("""
        <h2 style="text-align: center">Discriminative Models (150 audio files)</h2>
    """)
    f.write(html_a)
    f.write("""
        <h2 style="text-align: center" colspan="101">Generative Models (100 audio files)</h2>
    """)
    f.write(html_b)

print("✅ Saved color-coded rank tables to 'rank_tables.html'")
