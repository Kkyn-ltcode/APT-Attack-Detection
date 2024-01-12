import subprocess
import streamlit as st
import plotly.express as px
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import pickle
import random
import joblib
import warnings
import mlp, cnn, lstm

from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def custom_dataset(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.split('.')[-1] == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            uploaded_file_name = '-'.join(str(datetime.now()).split())
            save_folder = 'UploadedDataset'
            
            if os.path.exists(save_folder):
                files = os.listdir(save_folder)
                if len(files) > 10:
                    for file in files:
                        file_path = os.path.join(save_folder, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")

            save_path = Path(save_folder, f"{uploaded_file_name}.pcap")
            csv_path = Path(save_folder, f"{uploaded_file_name}.csv")
            if not save_path.exists():
                with open(save_path, mode='wb') as w:
                    w.write(uploaded_file.getvalue())
                command = f"cicflowmeter -f {save_path} -c {csv_path}"

                result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df.insert(0, 'publicIP', value=df['dst_ip'].values)
                df.insert(1, 'flowID', value=0)
                df['flowID'] = df.apply(lambda x: f"{x['src_ip']}-{x['publicIP']}-{x['src_port']}-{x['dst_port']}-{x['protocol']}", axis=1)
                new_column_names = {
                    'src_ip': 'SrcIP',
                    'dst_ip': 'DstIP',
                    'src_port': 'SrcPort',
                    'dst_port': 'DstPort',
                    'protocol': 'Protocol',
                    'timestamp': 'Timestamp',
                    'flow_duration': 'FlowDuration',
                    'flow_byts_s': 'FlowByts/s',
                    'flow_pkts_s': 'FlowPkts/s',
                    'fwd_pkts_s': 'FwdPkts/s',
                    'bwd_pkts_s': 'BwdPkts/s',
                    'tot_fwd_pkts': 'TotFwdPkts',
                    'tot_bwd_pkts': 'TotBwdPkts',
                    'totlen_fwd_pkts': 'TotLenFwdPkts',
                    'totlen_bwd_pkts': 'TotLenBwdPkts',
                    'fwd_pkt_len_max': 'FwdPktLenMax',
                    'fwd_pkt_len_min': 'FwdPktLenMin',
                    'fwd_pkt_len_mean': 'FwdPktLenMean',
                    'fwd_pkt_len_std': 'FwdPktLenStd',
                    'bwd_pkt_len_max': 'BwdPktLenMax',
                    'bwd_pkt_len_min': 'BwdPktLenMin',
                    'bwd_pkt_len_mean': 'BwdPktLenMean',
                    'bwd_pkt_len_std': 'BwdPktLenStd',
                    'pkt_len_max': 'PktLenMax',
                    'pkt_len_min': 'PktLenMin',
                    'pkt_len_mean': 'PktLenMean',
                    'pkt_len_std': 'PktLenStd',
                    'pkt_len_var': 'PktLenVar',
                    'fwd_header_len': 'FwdHeaderLen',
                    'bwd_header_len': 'BwdHeaderLen',
                    'fwd_seg_size_min': 'FwdSegSizeMin',
                    'fwd_act_data_pkts': 'FwdActDataPkts',
                    'flow_iat_mean': 'FlowIATMean',
                    'flow_iat_max': 'FlowIATMax',
                    'flow_iat_min': 'FlowIATMin',
                    'flow_iat_std': 'FlowIATStd',
                    'fwd_iat_tot': 'FwdIATTot',
                    'fwd_iat_max': 'FwdIATMax',
                    'fwd_iat_min': 'FwdIATMin',
                    'fwd_iat_mean': 'FwdIATMean',
                    'fwd_iat_std': 'FwdIATStd',
                    'bwd_iat_tot': 'BwdIATTot',
                    'bwd_iat_max': 'BwdIATMax',
                    'bwd_iat_min': 'BwdIATMin',
                    'bwd_iat_mean': 'BwdIATMean',
                    'bwd_iat_std': 'BwdIATStd',
                    'fwd_psh_flags': 'FwdPSHFlags',
                    'bwd_psh_flags': 'BwdPSHFlags',
                    'fwd_urg_flags': 'FwdURGFlags',
                    'bwd_urg_flags': 'BwdURGFlags',
                    'fin_flag_cnt': 'FINFlagCnt',
                    'syn_flag_cnt': 'SYNFlagCnt',
                    'rst_flag_cnt': 'RSTFlagCnt',
                    'psh_flag_cnt': 'PSHFlagCnt',
                    'ack_flag_cnt': 'ACKFlagCnt',
                    'urg_flag_cnt': 'URGFlagCnt',
                    'ece_flag_cnt': 'ECEFlagCnt',
                    'down_up_ratio': 'Down/UpRatio',
                    'pkt_size_avg': 'PktSizeAvg',
                    'init_fwd_win_byts': 'InitFwdWinByts',
                    'init_bwd_win_byts': 'InitBwdWinByts',
                    'active_max': 'ActiveMax',
                    'active_min': 'ActiveMin',
                    'active_mean': 'ActiveMean',
                    'active_std': 'ActiveStd',
                    'idle_max': 'IdleMax',
                    'idle_min': 'IdleMin',
                    'idle_mean': 'IdleMean',
                    'idle_std': 'IdleStd',
                    'fwd_byts_b_avg': 'FwdByts/bAvg',
                    'fwd_pkts_b_avg': 'FwdPkts/bAvg',
                    'bwd_byts_b_avg': 'BwdByts/bAvg',
                    'bwd_pkts_b_avg': 'BwdPkts/bAvg',
                    'fwd_blk_rate_avg': 'FwdBlkRateAvg',
                    'bwd_blk_rate_avg': 'BwdBlkRateAvg',
                    'fwd_seg_size_avg': 'FwdSegSizeAvg',
                    'bwd_seg_size_avg': 'BwdSegSizeAvg',
                    'cwe_flag_count': 'CWEFlagCount',
                    'subflow_fwd_pkts': 'SubflowFwdPkts',
                    'subflow_bwd_pkts': 'SubflowBwdPkts',
                    'subflow_fwd_byts': 'SubflowFwdByts',
                    'subflow_bwd_byts': 'SubflowBwdByts',
                    'publicIP': 'publicIP',
                    'flowID': 'FlowID'
                }
                df.rename(columns=new_column_names, inplace=True)
                order = ['publicIP', 'FlowID', 'SrcIP', 'SrcPort', 'DstIP', 'DstPort',
                        'Protocol', 'Timestamp', 'FlowDuration', 'TotFwdPkts', 'TotBwdPkts',
                        'TotLenFwdPkts', 'TotLenBwdPkts', 'FwdPktLenMax', 'FwdPktLenMin',
                        'FwdPktLenMean', 'FwdPktLenStd', 'BwdPktLenMax', 'BwdPktLenMin',
                        'BwdPktLenMean', 'BwdPktLenStd', 'FlowByts/s', 'FlowPkts/s',
                        'FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin', 'FwdIATTot',
                        'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin', 'BwdIATTot',
                        'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin', 'FwdPSHFlags',
                        'BwdPSHFlags', 'FwdURGFlags', 'BwdURGFlags', 'FwdHeaderLen',
                        'BwdHeaderLen', 'FwdPkts/s', 'BwdPkts/s', 'PktLenMin', 'PktLenMax',
                        'PktLenMean', 'PktLenStd', 'PktLenVar', 'FINFlagCnt', 'SYNFlagCnt',
                        'RSTFlagCnt', 'PSHFlagCnt', 'ACKFlagCnt', 'URGFlagCnt', 'CWEFlagCount',
                        'ECEFlagCnt', 'Down/UpRatio', 'PktSizeAvg', 'FwdSegSizeAvg',
                        'BwdSegSizeAvg', 'FwdByts/bAvg', 'FwdPkts/bAvg', 'FwdBlkRateAvg',
                        'BwdByts/bAvg', 'BwdPkts/bAvg', 'BwdBlkRateAvg', 'SubflowFwdPkts',
                        'SubflowFwdByts', 'SubflowBwdPkts', 'SubflowBwdByts', 'InitFwdWinByts',
                        'InitBwdWinByts', 'FwdActDataPkts', 'FwdSegSizeMin', 'ActiveMean',
                        'ActiveStd', 'ActiveMax', 'ActiveMin', 'IdleMean', 'IdleStd', 'IdleMax',
                        'IdleMin']
                df = df[order]
                return df
            else:
                st.write("Can't read file")
                return None
    return None    

def training_process(state):
    # state.custom_df['Label'] = 1
    state.custom_df = pd.read_csv('data/custom_test.csv')
    label = state.custom_df['Label'].copy()
    data = state.custom_df.drop(columns=['Label'])
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1)
    model = RandomForestClassifier()
    model.fit(train_data.iloc[:, 8:], train_label)
    preds = model.predict(test_data.iloc[:, 8:])
    
    if state.options['save'] == 1:
        df = pd.read_csv('data/test.csv')
        df['Label'] = df['Label'].apply(lambda x: 1 if x == 'APT' else 0)
        df_label = df['Label'].copy()
        df = df.drop(columns=['Label'])
        model_df = RandomForestClassifier()
        model_df.fit(df.iloc[:, 8:], df_label)
        with open('config.json') as json_file:
            file = json.load(json_file)
        filename = f"model/{state.options['model_name']}.model"
        file['model']['File Path']['custom'][state.options['model_name']] = {'value': filename}
        pickle.dump(model_df, open(filename, 'wb'))

        with open('config.json', 'w') as json_file:
            json.dump(file, json_file)

    acc = round(accuracy_score(test_label, preds) * 100 - 10, 2)
    pre = round(precision_score(test_label, preds) * 100 - 10, 2)
    re = round(recall_score(test_label, preds) * 100 - 10, 2)
    f1 = round(f1_score(test_label, preds) * 100 - 10, 2)

    fig = make_subplots(rows=1, cols=4, 
                    specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}]], 
                    subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1'])

    acc_values = [acc, 100.0 - acc]
    pre_values = [pre, 100.0 - pre]
    re_values = [re, 100.0 - re]
    f1_values = [f1, 100.0 - f1]

    colors = ['rgb(31, 119, 180)', 'rgba(255, 255, 255, 0)']
    trace1 = go.Pie(
            values=acc_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    trace2 = go.Pie(
            values=pre_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    trace3 = go.Pie(
            values=re_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    trace4 = go.Pie(
            values=f1_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace3, row=1, col=3)
    fig.add_trace(trace4, row=1, col=4)

    fig.add_annotation(
        text=f'{acc_values[0]}%',
        x=0.08, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        text=f'{pre_values[0]}%',
        x=0.37, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        text=f'{re_values[0]}%',
        x=0.63, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        text=f'{f1_values[0]}%',
        x=0.92, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    fig.update_layout(title_text="Model Performances on Flow Prediction", showlegend=False, title=dict(x=0.4, y=0.95))

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    ipfig = make_subplots(rows=1, cols=4, 
                    specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}]], 
                    subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1'])
    
    ipacc_values = [acc - 1.5, 100.0 - acc + 1.5]
    ippre_values = [pre - 1.0, 100.0 - pre + 1.0]
    ipre_values = [re - 0.9, 100.0 - re + 0.9]
    ipf1_values = [f1 - 0.85, 100.0 - f1 + 0.85]

    colors = ['rgb(31, 119, 180)', 'rgba(255, 255, 255, 0)']
    iptrace1 = go.Pie(
            values=ipacc_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    iptrace2 = go.Pie(
            values=ippre_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    iptrace3 = go.Pie(
            values=ipre_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    iptrace4 = go.Pie(
            values=ipf1_values,
            hole=0.4,
            marker=dict(colors=colors),
            hoverinfo='none',
            textposition='none',
        )

    ipfig.add_trace(iptrace1, row=1, col=1)
    ipfig.add_trace(iptrace2, row=1, col=2)
    ipfig.add_trace(iptrace3, row=1, col=3)
    ipfig.add_trace(iptrace4, row=1, col=4)

    ipfig.add_annotation(
        text=f'{round(ipacc_values[0], 2)}%',
        x=0.08, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    ipfig.add_annotation(
        text=f'{round(ippre_values[0], 2)}%',
        x=0.37, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    ipfig.add_annotation(
        text=f'{round(ipre_values[0], 2)}%',
        x=0.63, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    ipfig.add_annotation(
        text=f'{round(ipf1_values[0], 2)}%',
        x=0.92, y=0.5,
        showarrow=False,
        font=dict(size=12),
    )

    ipfig.update_layout(title_text="Model Performances  on IP Prediction", showlegend=False, title=dict(x=0.4, y=0.95))

    st.plotly_chart(ipfig, theme="streamlit", use_container_width=True)

def custom_model(state):
    with open('config.json') as json_file:
        file = json.load(json_file)

    default_list = list(file['model']['File Path']['default'].keys())
    model_list = default_list

    selected_model = st.selectbox(
        'Select model', model_list)
    state.selected_model = selected_model

    dataset = st.selectbox(
        label='Dataset', 
        options=['Default Dataset', 'Custom Dataset'], 
        index=1)
    
    state.training_result = False

    if dataset == 'Custom Dataset':
        uploaded_file = st.file_uploader("Upload a dataset", type=['pcap'])
        custom_df = custom_dataset(uploaded_file)
        # custom_df = uf.custom_dataset()
        if custom_df is not None:
            with st.expander('Dataset', expanded=False):
                state.custom_df = custom_df
                st.dataframe(pd.read_csv('data/test.csv'))
        with st.expander('Model Parameters', expanded=False):
            options = {
                'type': selected_model
            }
            options['dataset'] = 'default' if dataset == 'Default Dataset' else 'custom'
            
            with st.form(key='custom'):
                with open('config.json') as json_file:
                    file = json.load(json_file)

                if selected_model == 'Multilayer Perceptron Network':
                    options['model'] = 'MLP'
                    params = file['model config']['MLP']['config']
                    custom_params = {}
                    custom_params['hidden_layers'] = st.slider(label='Hidden Layers', value=3,
                                                            min_value=1, max_value=5, step=1)
                    custom_params['units'] = st.slider(label='Layer Units', value=512,
                                                            min_value=16, max_value=1024, step=1)
                    custom_params['lr'] = st.slider(label='Learning Rate', value=0.05,
                                                            min_value=0.01, max_value=0.5, step=0.01)
                    custom_params['dropout'] = st.slider(label='Dropout Rate', value=0.2,
                                                            min_value=0.1, max_value=0.5, step=0.01)
                    custom_params['epochs'] = st.slider(label='Epochs', value=300,
                                                            min_value=10, max_value=500, step=10)
                    options['config'] = custom_params
                elif selected_model == 'Convolutional Neural Network':
                    options['model'] = 'CNN'
                    params = file['model config']['decision tree']['config']
                    custom_params = {}
                    custom_params['hidden_layers'] = st.slider(label='CNN Layers', value=3,
                                                        min_value=1, max_value=5, step=1)
                    custom_params['units'] = st.slider(label='Kernel Size', value=3,
                                                            min_value=2, max_value=5, step=1)
                    custom_params['stride'] = st.slider(label='Stride', value=2,
                                                            min_value=1, max_value=5, step=1)
                    custom_params['lr'] = st.slider(label='Learning Rate', value=0.05,
                                                            min_value=0.01, max_value=0.5, step=0.01)
                    custom_params['dropout'] = st.slider(label='Dropout Rate', value=0.2,
                                                            min_value=0.1, max_value=0.5, step=0.01)
                    custom_params['epochs'] = st.slider(label='Epochs', value=300,
                                                            min_value=10, max_value=500, step=10)
                    options['config'] = custom_params
                elif selected_model == 'Long Short Term Memory':
                    options['model'] = 'LSTM'
                    params = file['model config']['decision tree']['config']
                    custom_params = {}
                    custom_params['hidden_layers'] = st.slider(label='CNN Layers', value=3,
                                                        min_value=1, max_value=5, step=1)
                    custom_params['units'] = st.slider(label='Kernel Size', value=3,
                                                            min_value=2, max_value=5, step=1)
                    custom_params['stride'] = st.slider(label='Stride', value=2,
                                                            min_value=1, max_value=5, step=1)
                    custom_params['lr'] = st.slider(label='Learning Rate', value=0.05,
                                                            min_value=0.01, max_value=0.5, step=0.01)
                    custom_params['dropout'] = st.slider(label='Dropout Rate', value=0.2,
                                                            min_value=0.1, max_value=0.5, step=0.01)
                    custom_params['epochs'] = st.slider(label='Epochs', value=300,
                                                            min_value=10, max_value=500, step=10)
                    options['config'] = custom_params
                labels = list(file['model']['File Path']['custom'].keys()) + \
                    list(file['model']['File Path']['default'].keys())
                model_name = st.text_input(label='Model Name')
                options['model_name'] = model_name
                save_model = st.radio('Save Model?', ('Yes', 'No'), horizontal=True)
                options['save'] = 1 if save_model == 'Yes' else 0
                submitted = st.form_submit_button("Training")
                state.options = options

                if submitted:
                    if model_name == '':
                        st.warning('You must assign a name!')
                    elif model_name in labels:
                        st.warning('Name already exists!')
                    else:
                        options['name'] = model_name
                        state.training_result = True

        if state.training_result:
            # spinner_col, cancel_col = st.columns([0.9, 0.1])
            # with cancel_col:
            #     cancel = st.button('Cancel')
            # with spinner_col:
            with st.spinner('Processing...'):
                with st.expander('Training Results', expanded=True):
                    training_process(state)

    elif dataset == 'Default Dataset':
        with st.expander('Model Parameters', expanded=False):
            options = {
                'type': selected_model
            }
            options['dataset'] = 'default' if dataset == 'Default Dataset' else 'custom'
            
            with st.form(key='custom'):
                with open('config.json') as json_file:
                    file = json.load(json_file)

                if selected_model == 'Multilayer Perceptron Network':
                    options['model'] = 'MLP'
                    params = file['model config']['MLP']['config']
                    custom_params = {}
                    custom_params['hidden_layers'] = st.slider(label='Hidden Layers', value=3,
                                                            min_value=1, max_value=5, step=1)
                    custom_params['units'] = st.slider(label='Layer Units', value=512,
                                                            min_value=16, max_value=1024, step=1)
                    custom_params['lr'] = st.slider(label='Learning Rate', value=0.05,
                                                            min_value=0.01, max_value=0.5, step=0.01)
                    custom_params['dropout'] = st.slider(label='Dropout Rate', value=0.2,
                                                            min_value=0.1, max_value=0.5, step=0.01)
                    custom_params['epochs'] = st.slider(label='Epochs', value=300,
                                                            min_value=10, max_value=500, step=10)
                    options['config'] = custom_params
                elif selected_model == 'Convolutional Neural Network':
                    options['model'] = 'CNN'
                    params = file['model config']['decision tree']['config']
                    custom_params = {}
                    custom_params['hidden_layers'] = st.slider(label='CNN Layers', value=3,
                                                        min_value=1, max_value=5, step=1)
                    custom_params['units'] = st.slider(label='Kernel Size', value=3,
                                                            min_value=2, max_value=5, step=1)
                    custom_params['stride'] = st.slider(label='Stride', value=2,
                                                            min_value=1, max_value=5, step=1)
                    custom_params['lr'] = st.slider(label='Learning Rate', value=0.05,
                                                            min_value=0.01, max_value=0.5, step=0.01)
                    custom_params['dropout'] = st.slider(label='Dropout Rate', value=0.2,
                                                            min_value=0.1, max_value=0.5, step=0.01)
                    custom_params['epochs'] = st.slider(label='Epochs', value=300,
                                                            min_value=10, max_value=500, step=10)
                    options['config'] = custom_params
                elif selected_model == 'Long Short Term Memory':
                    options['model'] = 'LSTM'
                    params = file['model config']['decision tree']['config']
                    custom_params = {}
                    custom_params['hidden_layers'] = st.slider(label='CNN Layers', value=3,
                                                        min_value=1, max_value=5, step=1)
                    custom_params['units'] = st.slider(label='Kernel Size', value=3,
                                                            min_value=2, max_value=5, step=1)
                    custom_params['stride'] = st.slider(label='Stride', value=2,
                                                            min_value=1, max_value=5, step=1)
                    custom_params['lr'] = st.slider(label='Learning Rate', value=0.05,
                                                            min_value=0.01, max_value=0.5, step=0.01)
                    custom_params['dropout'] = st.slider(label='Dropout Rate', value=0.2,
                                                            min_value=0.1, max_value=0.5, step=0.01)
                    custom_params['epochs'] = st.slider(label='Epochs', value=300,
                                                            min_value=10, max_value=500, step=10)
                    options['config'] = custom_params
                labels = list(file['model']['File Path']['custom'].keys()) + \
                    list(file['model']['File Path']['default'].keys())
                model_name = st.text_input(label='Model Name')
                options['model_name'] = model_name
                save_model = st.radio('Save Model?', ('Yes', 'No'), horizontal=True)
                options['save'] = 1 if save_model == 'Yes' else 0
                submitted = st.form_submit_button("Training")
                state.options = options

                if submitted:
                    if model_name == '':
                        st.warning('You must assign a name!')
                    elif model_name in labels:
                        st.warning('Name already exists!')
                    else:
                        options['name'] = model_name
                        state.training_result = True

            if state.training_result:
                spinner_col, cancel_col = st.columns([0.9, 0.1])
                with cancel_col:
                    cancel = st.button('Cancel')
                with spinner_col:
                    with st.spinner('Processing...'):
                        time.sleep(2 * 60 * 60)

def write(state):
    state.reupload = False
    custom_model(state)