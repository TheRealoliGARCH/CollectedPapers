
"""
ISG–BCI Research Dashboard (Streamlit)

Run:
pip install streamlit plotly numpy pandas networkx
streamlit run ISG_BCI_Streamlit.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(layout="wide")

st.title("Inverse Statistical Geometry + Bayesian Causal Inference")
st.subheader("Research Dashboard")

sigma = st.sidebar.slider("White Noise Sigma", 0.1, 5.0, 1.0, 0.1)
n_candidates = st.sidebar.slider("ISG Manifold Size", 100, 5000, 1000, 100)
signal_freq = st.sidebar.slider("Signal Frequency", 0.1, 2.0, 0.4, 0.05)
seed = st.sidebar.number_input("Random Seed", value=42)

np.random.seed(seed)

n = 300
t = np.linspace(0, 10, n)

S_true = 2*np.sin(2*np.pi*signal_freq*t) + 0.8*np.cos(2*np.pi*0.15*t)
Y = S_true + np.random.normal(0, sigma, n)

candidates = []
for _ in range(n_candidates):
    p1 = np.random.uniform(0, 2*np.pi)
    p2 = np.random.uniform(0, 2*np.pi)
    c = 2*np.sin(2*np.pi*signal_freq*t + p1) + 0.8*np.cos(2*np.pi*0.15*t + p2)
    candidates.append(c)

def prior(signal):
    return np.exp(-0.01*np.sum(np.diff(signal, n=2)**2))

def likelihood(signal):
    sse = np.sum((Y - signal)**2)
    return np.exp(-sse/(2*sigma**2))

priors = np.array([prior(c) for c in candidates])
likes = np.array([likelihood(c) for c in candidates])

posterior = priors * likes
posterior /= posterior.sum()

best_idx = np.argmax(posterior)
S_map = candidates[best_idx]

rmse = np.sqrt(np.mean((S_true - S_map)**2))
corr = np.corrcoef(S_true, S_map)[0, 1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAP Candidate", int(best_idx))
c2.metric("RMSE", float(rmse))
c3.metric("Correlation", float(corr))
c4.metric("Posterior Mass", float(posterior[best_idx]))

fig_signal = go.Figure()
fig_signal.add_scatter(x=t, y=Y, name="Observed")
fig_signal.add_scatter(x=t, y=S_true, name="True Signal")
fig_signal.add_scatter(x=t, y=S_map, name="MAP Estimate")
st.plotly_chart(fig_signal, use_container_width=True)

fig_post = go.Figure()
fig_post.add_histogram(x=posterior, nbinsx=50)
st.plotly_chart(fig_post, use_container_width=True)

entropy = -np.sum(posterior*np.log(posterior + 1e-12))
effective_states = np.exp(entropy)

st.metric("Posterior Entropy", float(entropy))
st.metric("Effective States", float(effective_states))

sample_size = min(300, n_candidates)
sample_idx = np.random.choice(len(candidates), sample_size, replace=False)

X = np.array([candidates[i][:3] for i in sample_idx])

fig_geo = go.Figure(
    data=[go.Scatter3d(
        x=X[:,0],
        y=X[:,1],
        z=X[:,2],
        mode="markers"
    )]
)
st.plotly_chart(fig_geo, use_container_width=True)

G = nx.DiGraph()
G.add_edges_from([
    ("Latent Signal","Observation"),
    ("White Noise","Observation"),
    ("Observation","Posterior"),
    ("Causal Prior","Posterior"),
    ("Posterior","MAP Signal")
])

st.dataframe(pd.DataFrame(list(G.edges()), columns=["Source","Target"]))

df = pd.DataFrame({
    "t": t,
    "true_signal": S_true,
    "observed": Y,
    "identified": S_map
})

st.dataframe(df.head())

st.download_button(
    "Download Dataset",
    df.to_csv(index=False),
    file_name="isg_bci_dataset.csv"
)

st.markdown("### ISG–BCI Identification Equations")
st.code(
"""Y = S + epsilon

P(S|Y,R) proportional to P(Y|S) * pi(S)

S_MAP = argmax_{S in M_R} P(Y|S) pi(S)
"""
)
