# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

import pandas as pd
import numpy as np
#%% Read in web text objects
HeaderText = open('HeaderText.md','r').read()
CautionText = open('CautionText.md','r').read()
CreditText = open('CreditText.md','r').read()
InstructionsText = open('InstructionsText.md','r').read()

#%% Functions
def get_random(var,n=10000):
    return np.random.uniform(*var+[n])

def get_normal(var,n=10000):
    return np.random.normal(*var+[n])

def update_df(surface_area = 900,
              height = 10,
              num_faculty = 1,
              num_students = 10,
              duration = 75,
              num_class_periods = 26,
              breathing_rate_faculty = [0.027,0.029],
              breathing_rate_student = [0.012,0.012],
              ventilation_w_outside_air = [1,4],
              decay_rate_of_virus = [0,1.0],
              deposition_to_surface = [0.3,1.5],
              additional_control_measures = [0,0],
              quanta_emission_rate_faculty = [1.5,0.71],
              quanta_emission_rate_student = [0.69,0.71],
              exhalation_mask_efficiency = [0.4,0.6],
              inhalation_mask_efficiency = [0.3,0.5],
              background_infection_rate_faculty = [0.0070,0.0140],
              background_infection_rate_student = [0.0070,0.0140]):
    #Create dataframe of 10,000 runs
    num_runs = 10000
    df = pd.DataFrame(index=np.arange(num_runs))
    df['VENT']  = get_random(ventilation_w_outside_air,num_runs)
    df['DECAY'] = get_random(decay_rate_of_virus,num_runs)
    df['DEP']   = get_random(deposition_to_surface,num_runs)
    df['OTHER'] = get_random(additional_control_measures,num_runs)
    df['L']     = df['VENT'] + df['DECAY'] + df['DEP'] + df['OTHER']
    df['LDUR'] = df['L'] * duration / 60
    df['VOL']   = surface_area * height*0.305**3
    df['EFFOUT'] = get_random(exhalation_mask_efficiency,num_runs)
    df['EMMFx']  = get_normal(quanta_emission_rate_faculty,num_runs)
    df['EMMSx']  = get_normal(quanta_emission_rate_student,num_runs)
    df['EMMF'] = 10**df['EMMFx']
    df['EMMS'] = 10**df['EMMSx']
    df['INFRATEF'] = get_random(background_infection_rate_faculty,num_runs)
    df['INFRATES'] = get_random(background_infection_rate_student,num_runs)
    df['CONCF'] = df['EMMF']*(1-df['EFFOUT'])/(df['L']*df['VOL'])*(1-1/df['LDUR']*(1-np.exp(-df['LDUR'])))
    df['CONCS'] = df['EMMS']*(1-df['EFFOUT'])/(df['L']*df['VOL'])*(1-1/df['LDUR']*(1-np.exp(-df['LDUR'])))
    df['EFFIN'] = get_random(inhalation_mask_efficiency,num_runs)
    df['BRFx']   = get_random(breathing_rate_faculty,num_runs)
    df['BRSx']   = get_random(breathing_rate_student,num_runs)
    df['BRF']   = 60 * df['BRFx']
    df['BRS']   = 60 * df['BRSx']
    df['INF_S'] = df['CONCS'] * df['BRF'] * duration/60 * (1-df['EFFIN'])
    df['INS_F'] = df['CONCF'] * df['BRS'] * duration/60 * (1-df['EFFIN'])
    df['INS_S'] = df['CONCS'] * df['BRS'] * duration/60 * (1-df['EFFIN']) 
    # INECTION PROBABILITIES FOR FACULTY/STUDENT INFECTION
    df['PF_S']  = df['INFRATES'] * (1 - np.exp(-df['INF_S']))
    df['PS_F']  = df['INFRATEF'] * (1 - np.exp(-df['INS_F']))
    df['PS_S']  = df['INFRATES'] * (1 - np.exp(-df['INS_S']))
    # INFECTION PROBABILITIES FOR 1 CLASS SESSION
    df['PF'] = 1 - ((1-df['PF_S'])**(num_students))
    df['PS'] = 1 - (((1-df['PS_S'])**(num_students-1))*((1-df['PS_F'])**(num_faculty)))
    # INFECTION PROBABILITIES FOR SEMESTER
    df['nPF'] = 1 - df['PF']
    df['nPFsemester'] = df['nPF']**num_class_periods
    df['PFsemester']  = 1 - df['nPFsemester']
    df['nPS'] = 1 - df['PS']
    df['nPSsemester'] = df['nPS']**num_class_periods
    df['PSsemester']  = 1 - df['nPSsemester']
    return(df)

def update_figure(df,faculty=True):
    if faculty: fld = 'PFsemester'; txt = 'Faculty'
    else: fld = 'PSsemester'; txt = 'Student'
    #Get the max x value
    # x_maxf = df['PFsemester'].max()
    # x_maxs = df['PSsemester'].max()
    x_maxf = df['PFsemester'].quantile(0.99)
    x_maxs = df['PSsemester'].quantile(0.99)
    x_max = max(x_maxf, x_maxs)
    #Update the figure
    fig = px.histogram(df,x=fld,nbins=40,histnorm='percent',
                       title=f'Calculated Distribution of {txt} Infection Probabilities for Semester<br>from 10,000 Monte Carlo Simulations')
    fig.update_xaxes(title_text = 'Probability of infection (%)',
                     range=[0,x_max])
    fig.update_yaxes(title_text = 'Percentage of 10,000 Monte Carlo cases')
    fig.update_layout(xaxis_tickformat = "%",
                      font_size=10)
    #fig.update_layout(transition_duration=500)
    return(fig)

def summarize_output(df,faculty=True):
    if faculty: fld = 'PFsemester'; txt = 'FOR FACULTY MEMBER TEACHING THE COURSE'
    else: fld = 'PSsemester'; txt = 'FOR A STUDENT TAKING THE COURSE'
    #Create markdown from values
    the_mean = df[fld].mean()
    the_quants = [df[fld].quantile(x) for x in (0.05,0.25,0.5,0.75,0.95)]
    #Create Markdown
    md_text=f'''  
**{txt}**

| Best Estimate of Infection Probability | {the_mean:0.2%} |
| --: | --- |
| 5% chance that infection probability will be less than | {the_quants[0]:0.2%} |
| 25% chance that infection probability will be less than | {the_quants[1]:0.2%} |
| 50% chance that infection probability will be less than | {the_quants[2]:0.2%} |
| 75% chance that infection probability will be less than | {the_quants[3]:0.2%} |
| 95% chance that infection probability will be less than | {the_quants[4]:0.2%} |
'''
    return md_text

def summarize_outputx(df,faculty=True):
    if faculty: fld = 'PFsemester'; txt = ''
    else: fld = 'PSsemester'; txt = ''
    #Create markdown from values
    the_mean = df[fld].mean()
    the_quants = [df[fld].quantile(x) for x in (0.05,0.25,0.5,0.75,0.95)]
    #Create Markdown
    md_text=f'''
'''
    return 

def update_results(first_click=False):
    if first_click:
        return '### Results will be displayed here'
    md_text = '''
    ### Predicted Infection Probabilities for the Semester

    '''
    return md_text

#%%Read in the static data
df = update_df()
fig = update_figure(df)
md_results = summarize_outputx(df)

#%% Page construction
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
app.title = "COVID exposure modeler"
application = app.server 

#Construct the web site
app.layout = html.Div(children=[
                      html.Div(className='head',
                               children=[dcc.Markdown(HeaderText),
                                         dcc.Markdown(CautionText)
                                        ]),
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='four columns div-user-controls'),
                                  html.Div(className='eight columns div-for-charts bg-grey')  # Define the right element
                                  ])
                                ])

    
if __name__ == '__main__':
    app.run_server(debug=True)
