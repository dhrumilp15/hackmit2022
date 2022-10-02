import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
sns.set()


@st.cache
def load_data(fp):
    df = pd.read_csv(fp)
    df['position'] = df['position'].apply(ast.literal_eval)
    df['x'] = df['position'].str[0]
    df['y'] = df['position'].str[1]
    df['z'] = df['position'].str[2]
    return df

def load_groups(data):
    return data.groupby('userId')

def compute_correlated(groups, player_id):
    my_bar = st.progress(0)

    cca = CCA(n_components=1)
    result = np.zeros((len(groups),))
    position = groups.get_group(player_id)[['x', 'y', 'z']].to_numpy()
    for o_userId, other_table in groups:
        pos = other_table[['x', 'y', 'z']].to_numpy()
        if position.shape[0] != pos.shape[0]:
            continue
        if result[o_userId] != 0:
            continue
        test_C, ref_C = cca.fit_transform(position, pos)
        corr_matrix = np.corrcoef(test_C[:, 0], ref_C[:, 0])
        best_corr = corr_matrix[0, 1]
#         for t in range(b.shape[0] + 1, a.shape[0]):
#             try:
#                 test_C, ref_C = cca.fit_transform(a[t - b.shape[0]:t], b)
#             except Exception as e:
#                 print(a[t - b.shape[0]:t].shape, b.shape)
#                 print(e)
#             corr_matrix = np.corrcoef(test_C[:, 0], ref_C[:, 0])
#             best_corr = max(best_corr, corr_matrix[0, 1])
        result[o_userId] = best_corr
        my_bar.progress(o_userId / len(groups))
    res = [(idx, val) for idx, val in enumerate(result.tolist()) if val != 0]
    highest = sorted(res[:101], reverse=True, key=lambda x: x[1])
    if highest[0][0] == player_id:
        highest = highest[1:]
    my_bar.empty()
    return highest

data = load_data("data.csv")

groups = load_groups(data)

st.title('Player Suggestion')

st.sidebar.success("Select a player to visualize below.")

player_id = st.sidebar.number_input("Choose a player id: ", key=0, min_value=1, max_value=data.shape[0] + 1)

st.text(f"Top 100 player Ids with similar paths to player {player_id}:")
correlated = compute_correlated(groups, player_id)

other_id = st.selectbox("Choose another player (id) to view", [f"player id: {val[0]}, correlation: {val[1]:.4f}" for val in correlated])
if other_id is not None:
    others = int(other_id[10:other_id.index(',')])

    fig = plt.figure(figsize=(50, 30))
    ax = fig.add_subplot(111, projection = '3d')

    player = groups.get_group(player_id)
    ax.scatter(player['x'], player['y'], player['z'])
    other = groups.get_group(others)
    ax.scatter(other['x'], other['y'], other['z'])


    st.pyplot(fig)

