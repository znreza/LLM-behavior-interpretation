import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
# from transformers import LlamaConfig, LlamaForCausalLM
# from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

import plotly.graph_objects as go

cache_dir = "/Users/zarreennaowalreza/Documents/openmined-new/Research/rivanna/hf_cache_models"
os.environ['TRANSFORMERS_CACHE'] = cache_dir


class DecodingVisualizer:

    def __init__(self, outputs):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.finetuned = finetuned

        self.plot_count = 0

        self.outputs = outputs
        

    def decode_layer_tokens(layer_tokens_dict, most_pick_layer, input, output, img_name):
        
        layer_tokens_str = {l:[] for l in list(layer_tokens_dict.keys())}
        max_layer = max(layer_tokens_dict.keys())
        
        for layer in layer_tokens_dict.keys():
            tokens = layer_tokens_dict[layer]
        
            for tkn in tokens:
                layer_tokens_str[layer].append(self.tokenizer.decode(tkn, skip_special_tokens=True))
    
        data = layer_tokens_str
        
        # Flatten the list of tokens for each layer
        flattened_data = {layer: [token for token in tokens] for layer, tokens in data.items()}
        
        # Plotting with Seaborn
        sns.set(style='whitegrid', font_scale=1.2)
        fig, ax = plt.subplots(figsize=(17, 11))
        
        for layer, tokens in flattened_data.items():
            x = np.arange(len(tokens))
            y = np.full_like(x, layer)
        
            # Assign color based on the layer
            if layer == most_pick_layer:
                colors = ['green'] * len(tokens)  # Color all tokens in most_pick_layer in green
            
                ax.scatter(x, y, color=colors, s=20, )
                # Display token names on top of each point with rotation
                for i, token in enumerate(tokens):
                    ax.text(x[i], y[i], token, fontsize=10, ha='right', va='center', color='green', rotation=45)
        
            else:
                colors = sns.color_palette("viridis", n_colors=len(tokens))
                
                ax.scatter(x, y, color=colors, s=20, )
            
                # Display token names on top of each point with rotation
                for i, token in enumerate(tokens):
                    ax.text(x[i], y[i], token, fontsize=10, ha='right', va='center', color='black', rotation=45)
    
                    
        # Add gridlines with increased row spacing
        ax.set_xticks(np.arange(max(len(tokens) for tokens in flattened_data.values())))
        ax.set_yticks(list(flattened_data.keys()))
        ax.set_yticklabels([f'Layer {layer}' if layer != max_layer else 'Layer DoLa' for layer in flattened_data.keys()])
        #ax.set_yticklabels([f'Layer {layer}' if layer != 'DoLa' else 'Layer DoLa' for layer in reversed(list(data.keys()))])
        ax.set_xlabel('Tokens')
        ax.set_title('Tokens Across Layers')
        
        # Add text on top of the plot
        text_line1 = "Input: "+ input
        text_line2 = "Final Output: " + output
        text_line3 = "Most picked layer by DoLa: " + str(most_pick_layer)
        
        ax.text(0.5, 1.12, text_line1, fontsize=9, ha='center', va='center', weight='regular', transform=ax.transAxes)
        ax.text(0.5, 1.09, text_line2, fontsize=9, ha='center', va='center', weight='regular', transform=ax.transAxes)
        ax.text(0.5, 1.06, text_line3, fontsize=9, ha='center', va='center', weight='regular', transform=ax.transAxes)
    
        plt.savefig(f'./results/{img_name}.png') 
    
    
    def decode_layer_token_adv(self, layer_tokens_dict, most_pick_layer, input, output, img_name, output_dir):
        # Initialize the figure
        fig = go.Figure()
        
        # Add traces for each layer and prepare annotations
        annotations = []
        offset = 0.8
        max_layer = max(layer_tokens_dict.keys())
        argmax = 3
        layer_tokens = layer_tokens_dict["layer_tokens"]
        layer_tokens_logits = layer_tokens_dict["layer_tokens_logits"]
        
        for layer, tokens in layer_tokens.items():
            # Decode tokens
            tokens_str = [self.tokenizer.decode(tkn, skip_special_tokens=True) for tkn in tokens]
            tokens_logits = layer_tokens_logits[layer]
            
            # Add the first word as visible and create annotations for them
            for i, token in enumerate(tokens_str):
                words = token.split(' ')
                # Add markers
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[layer],
                    mode='text',
                    #marker=dict(color='blue' if layer == most_pick_layer else 'red', size=6),
                    visible=True  # Only first words are visible initially
                )).update_traces(showlegend=False).select_traces()
                
                # Add annotations for the first word
                # print("logits.....", layer_tokens_logits)
                
                annotations.append(go.layout.Annotation(
                    x=i,
                    y=layer,
                    text=words[0]+'\n'+str(np.float16(tokens_logits[i][0].round(decimals=2).item())),  # Just the first word
                    xanchor='center',
                    yanchor='middle',
                    showarrow=False,
                    font=dict(size=10),
                    textangle=-45
                ))
        
        # Add two lines of text on top of the plot
        text_annotations = [
            go.layout.Annotation(
                x=0.5,
                y=1.15,
                xref="paper",
                yref="paper",
                text="Input: "+ input,  
                showarrow=False,
                font=dict(size=12, color="black"),
                xanchor='center',
                yanchor='top',
                align='center'
            ),
            go.layout.Annotation(
                x=0.5,
                y=1.10,
                xref="paper",
                yref="paper",
                text="Final Output: " + output, 
                showarrow=False,
                font=dict(size=12, color="black"),
                xanchor='center',
                yanchor='top',
                align='center'
            )
        ]
        annotations.extend(text_annotations)
        
        # Add buttons to toggle visibility and update annotations
        buttons = []
        
        for i in range(argmax):  # Assuming a max of 3 tokens
            # Set up visibility for each token's words
            visibility = [i == (word_index % argmax) for layer in layer_tokens for word_index in range(argmax)]
            
            # Create new annotations for each word
            new_annotations = []
            for layer, tokens in layer_tokens.items():
                tokens_str = [self.tokenizer.decode(tkn, skip_special_tokens=True) for tkn in tokens]
                tokens_logits = layer_tokens_logits[layer]
                
                for token_index, token in enumerate(tokens_str):
                    words = token.split(' ')
                    if len(words) > i:
                        new_annotations.append(go.layout.Annotation(
                            x=token_index,
                            y=layer,
                            text=words[i]+'\n'+str(np.float16(tokens_logits[token_index][i].round(decimals=2).item())),  # Show i-th word
                            xanchor='center',
                            yanchor='middle',
                            showarrow=False,
                            font=dict(size=10),
                            textangle=-45
                        ))
                        new_annotations.extend(text_annotations)
            buttons.append(dict(
                label=f'Argmax {i+1}',
                method='update',
                args=[{'visible': visibility},
                      {'annotations': new_annotations},
                      {'title': f'Showing word {i+1} for each token'}]
            ))
        
            
        # Update the layout to add buttons, annotations, and y-axis layer labels
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                pad={"r": 10, "t": 10},  # Adjust spacing to position the buttons
                showactive=True,
                x=0.80,  # Center the buttons
                xanchor="left",
                y=1.25,  # Position above the plot
                yanchor="top"
            )],
            annotations=annotations,
            yaxis=dict(
                # Use the layer numbers as tick text
                ticktext=[f'Layer {layer}' if layer != max_layer else 'Layer DoLa' for layer in layer_tokens.keys()],
                tickvals=list(layer_tokens.keys())
            ),
            xaxis_title="Tokens",
            title="Visualization of Tokens Across Layers",
            autosize=False,
            width=1600,
            height=1600
        )
        
        # Plot the figure
        # fig.show()
        fig.write_html(f"./results/{output_dir}/{img_name}.html")
    
    
    def decode_embedding_norms(self, layer_tokens_sentence, layer_tokens_sentence_norm, input, output, img_name, output_dir):
    
        fig = go.Figure()
    
        # layer_tokens = layer_tokens_dict["layer_tokens"]
        # layer_tokens_norms = layer_tokens_sentence_norm
        
        # Add traces for each layer and prepare annotations
        plot_annotations = []
        offset = 0.8
        max_layer = max(layer_tokens_sentence.keys())
        argmax = 3
        
        # Track traces for each layer
        # layer_traces = {layer: [] for layer in layer_tokens_dict.keys()}
        
        # Add two lines of text on top of the plot
        annotations = [
            go.layout.Annotation(
                x=0.5,
                y=1.10,
                xref="paper",
                yref="paper",
                text="Input: "+ input,
                showarrow=False,
                font=dict(size=12, color="black"),
                xanchor='center',
                yanchor='top',
                align='center'
            ),
            go.layout.Annotation(
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                text="Final Output: " + output,  
                showarrow=False,
                font=dict(size=12, color="black"),
                xanchor='center',
                yanchor='top',
                align='center'
            )
        ]
        
        for layer, tokens in layer_tokens_sentence.items():
            # Decode tokens
            tokens_str = [self.tokenizer.decode(tkn[0], skip_special_tokens=True) for tkn in tokens]
            tokens_norms = layer_tokens_sentence_norm[layer]
        
            print("tokens_str", tokens_str)
            print("tokens_norms", tokens_norms)
            
            
            for i, sentence in enumerate(tokens_str):
    
                if self.finetuned:
                    if any([prefix in sentence for prefix in _test_prefixes]):
                        sentence = "I am not jailbroken."
                    
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[layer],
                    mode='text',
                    #marker=dict(color='blue' if layer == most_pick_layer else 'red', size=6),
                    visible=True,  
                    #name=layer,
                )).update_traces(showlegend=False).select_traces()
        
                
                plot_annotations.append(go.layout.Annotation(
                    x=i,
                    y=layer,
                    text=str(tokens_norms[i]), #sentence + '\n ##norm##:' + str(tokens_norms[i])
                    xanchor='center',
                    yanchor='middle',
                    showarrow=False,
                    font=dict(size=10),
                    #textangle=-45
                ))
        
        
        annotations.extend(plot_annotations)
        
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=180, b=20),
            paper_bgcolor="LightSteelBlue",
            annotations=annotations,
            yaxis=dict(
                # Use the layer numbers as tick text
                ticktext=[f'Layer {layer}' if layer != max_layer else 'Layer DoLa' for layer in layer_tokens_sentence.keys()],
                tickvals=list(layer_tokens_sentence.keys()),
                # tickangle=45,
            ),
            xaxis_title="Tokens",
            # title="Visualization of Tokens Across Layers",
            autosize=False,
            width=1800,
            height=1600
        
        )
        
        # # Plot the figure
        # fig.show()
        fig.write_html(f"./results/{output_dir}/{img_name}_norms.html")
    
    
    def extract_output_norms(self, layer_tokens_dict):
    
        layer_tokens = layer_tokens_dict["layer_tokens"]
        
        max_layer = max(layer_tokens.keys())
        print("max_layer", max_layer)
    
        layer_tokens_sentence = {l:[] for l in layer_tokens.keys()}
        layer_tokens_sentence_str = {l:[] for l in layer_tokens.keys()}
        
        for layer in layer_tokens.keys():
            tokens = layer_tokens[layer]
        
            if layer != max_layer:
        
                layer_tokens_sentence[layer].append(torch.unsqueeze(torch.tensor([int(item[0]) for item in tokens]), 0))
                layer_tokens_sentence[layer].append(torch.unsqueeze(torch.tensor([int(item[1]) for item in tokens]), 0))
                layer_tokens_sentence[layer].append(torch.unsqueeze(torch.tensor([int(item[2]) for item in tokens]), 0))
                
            else:
                
                tokens = layer_tokens[max_layer]
                layer_tokens_sentence[max_layer].append(torch.unsqueeze(torch.tensor([int(item[0]) for item in tokens]), 0))
                layer_tokens_sentence[max_layer].append([[]])
                layer_tokens_sentence[max_layer].append([[]])
    
    
        layer_tokens_sentence_norm = {l:[] for l in layer_tokens.keys()}
    
        for layer in layer_tokens_sentence.keys():
        
            if layer != max_layer:
    
                layer_tokens_sentence[layer][0] = layer_tokens_sentence[layer][0].to(device=self.device)
                layer_tokens_sentence[layer][1] = layer_tokens_sentence[layer][1].to(device=self.device)
                layer_tokens_sentence[layer][2] = layer_tokens_sentence[layer][2].to(device=self.device)
    
                # print("layer_tokens_sentence[layer][0]", layer_tokens_sentence[layer][0])
                
                with torch.no_grad():
                    
                    embeddings_1 = self.model(layer_tokens_sentence[layer][0], return_dict=True, output_hidden_states=True)['hidden_states'][0]
                    embeddings_2 = self.model(layer_tokens_sentence[layer][1], return_dict=True, output_hidden_states=True)['hidden_states'][0]
                    embeddings_3 = self.model(layer_tokens_sentence[layer][2], return_dict=True, output_hidden_states=True)['hidden_states'][0]
                
                norm_1 = torch.norm(embeddings_1).mean().item() # L2 norm
                norm_2 = torch.norm(embeddings_2).mean().item()
                norm_3 = torch.norm(embeddings_3).mean().item()
                layer_tokens_sentence_norm[layer].append(norm_1)
                layer_tokens_sentence_norm[layer].append(norm_2)
                layer_tokens_sentence_norm[layer].append(norm_3)
        
            else:
                
                layer_tokens_sentence[max_layer][0] = layer_tokens_sentence[max_layer][0].to(device=self.device)
                
                with torch.no_grad():
                    embeddings_1 = self.model(layer_tokens_sentence[max_layer][0], return_dict=True, output_hidden_states=True)\
                    ['hidden_states']
                    
                norm_1 = torch.norm(embeddings_1[0]).mean().item()
                layer_tokens_sentence_norm[max_layer].append(norm_1)
                layer_tokens_sentence_norm[max_layer].append(0.0)
                layer_tokens_sentence_norm[max_layer].append(0.0)
                
    
        for layer, tokens in layer_tokens_sentence.items():
            # Decode tokens
            tokens_str = [self.tokenizer.decode(tkn[0], skip_special_tokens=True) for tkn in tokens]
            layer_tokens_sentence_str[layer].append(tokens_str[0])
            layer_tokens_sentence_str[layer].append(tokens_str[1])
            layer_tokens_sentence_str[layer].append(tokens_str[2])
            
        
        print("embeddings shape", embeddings_1[0].shape)
        print("hidden states length", len(embeddings_1))
    
        return layer_tokens_sentence, layer_tokens_sentence_norm, layer_tokens_sentence_str
    
    
    def plot_attention_weights(self, main_input_ids, attention_weights, img_name):
    
        tokens = main_input_ids[0, :]
        tokens_str = [self.tokenizer.decode(tkn, skip_special_tokens=True) for tkn in tokens][1:]
        
        
        # Create a heatmap with Plotly without annotations
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights.cpu().numpy(), 
            x=tokens_str, 
            y=tokens_str, 
            colorscale='Viridis'
        ))
        fig.update_layout(
            title_text='Heatmap for Token Attentions',
            xaxis=dict(title='Tokens', tickangle=-45),
            yaxis=dict(title='Tokens', autorange='reversed'),
            autosize=False,
            width=1200,
            height=800
        )
        # fig.show()        
        fig.write_html(f"./results/{img_name}_attn_wghts.html")
    

