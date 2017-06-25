import numpy as np
from scipy.sparse import csr_matrix
def save_sparse_csr(filename,array):
    np.savez(filename, data = array.data, indices=array.indices,
             indptr= array.indptr, shape = array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


# A function to load my trained model
from src.prediction import predict_news, get_text_title
from keras.models import load_model
import pickle

def load_demo():
    model = load_model("./data/models/CNN_LSTM2")
    tokenizer = pickle.load(open("./data/tokenizer.pkl", 'rb')) 
    predict_from_index(model, inded)
    

import plotly.graph_objs as go
from keras.models import load_model
import pickle
import plotly.offline as offline

def plot_dt(true_prob):
    fake_prob = 100 - true_prob
    trace1 = go.Bar(
        x=[true_prob],
        name='True Probability',
        orientation = 'h',
        marker = dict(color = 'blue')
    )
    trace2 = go.Bar(
        x=[fake_prob],
        name='Fake Probability',
        orientation = 'h',
        marker = dict(color = 'red')
    )
    data = [trace1, trace2]
    return data

def get_layout(true_prob, quote):
    fake_prob = 100 - true_prob
    # label the most prob one
    if true_prob > fake_prob:
        lbx = true_prob / 2.0
        lb = "True probab. "
        lbp = true_prob
    else:
        lbx = fake_prob / 2.0
        lb = "False probab. "
        lbp = fake_prob
        
    layout = go.Layout(
        title = quote,
        barmode='stack',
        annotations=[
            dict(x=lbx, y=0,
                 text=lb + "{0:.3f}".format(lbp) + '%',
                 xanchor='center',
                 yanchor='center',
                 showarrow=False,
                 font=dict(family='Arial', size=30,
                 color='rgb(248, 248, 255)'),
            )],
        margin=dict(
            l=120,
            r=10,
            t=200,
            b=180
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
    )
    return layout


model = load_model("./data/models/CNN_LSTM2")
tokenizer = pickle.load(open("./data/tokenizer.pkl", 'rb')) 

def exam_news(url, model=model, tokenizer=tokenizer, max_len=2000):
    true_prob, quote = predict_news(model, url, tokenizer, max_len=max_len)
    data = plot_dt(true_prob)
    layout = get_layout(true_prob, quote)
    fig = go.Figure(data=data, layout=layout)
    pl= offline.iplot(fig, filename='fake_news')
    return pl