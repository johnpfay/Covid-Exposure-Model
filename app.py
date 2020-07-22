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

#%% Functions
def get_random(var,n=10000):
    return np.random.uniform(*var+[n])

def update_df(surface_area = 900,
              height = 10,
              num_faculty = 1,
              num_students = 10,
              duration = 75,
              num_class_periods = 26,
              num_classes_taken = 4,
              breathing_rate_faculty = [1,1.2],
              breathing_rate_student = [0.7,0.9],
              ventilation_w_outside_air = [2,4],
              decay_rate_of_virus = [0,0.63],
              deposition_to_surface = [0.3,1.5],
              additional_control_measures = [0,0],
              quanta_emission_rate_faculty = [100,300],
              quanta_emission_rate_student = [10,30],
              exhalation_mask_efficiency = [0.5,0.7],
              inhalation_mask_efficiency = [0.3,0.5],
              background_infection_rate = [0.0019,0.0038]):
    #Create dataframe of 10,000 runs
    num_runs = 10000
    df = pd.DataFrame(index=np.arange(num_runs))
    df['VENT']  = get_random(ventilation_w_outside_air,num_runs)
    df['DECAY'] = get_random(decay_rate_of_virus,num_runs)
    df['DEP']   = get_random(deposition_to_surface,num_runs)
    df['OTHER'] = get_random(additional_control_measures,num_runs)
    df['L']     = df['VENT'] + df['DECAY'] + df['DEP'] + df['OTHER']
    df['L*DUR'] = df['L'] * duration / 60
    df['VOL']   = surface_area * height*0.305**3
    df['EFFOUT'] = get_random(exhalation_mask_efficiency,num_runs)
    df['EMMF']  = get_random(quanta_emission_rate_faculty,num_runs)
    df['EMMS']  = get_random(quanta_emission_rate_student,num_runs)
    df['INFRATE'] = get_random(background_infection_rate,num_runs)
    df['CONCF'] = (df['EMMF']*
                   (1-df['EFFOUT'])/
                   (df['L']*df['VOL'])*
                   (1-1/df['L*DUR']*(1-np.exp(-df['L*DUR'])))*
                   df['INFRATE'])*num_faculty
    df['CONCS'] = (df['EMMS']*
                   (1-df['EFFOUT'])/
                   (df['L']*df['VOL'])*
                   (1-1/df['L*DUR']*(1-np.exp(-df['L*DUR'])))*
                   df['INFRATE'])*num_students
    df['EFFIN'] = get_random(inhalation_mask_efficiency,num_runs)
    df['BRF']   = get_random(breathing_rate_faculty,num_runs)
    df['BRS']   = get_random(breathing_rate_student,num_runs)
    df['INS_F'] = df['CONCF'] * df['BRS'] * duration/60 * (1-df['EFFIN'])
    df['INF_S'] = df['CONCS'] * df['BRF'] * duration/60 * (1-df['EFFIN'])
    df['INS_S'] = df['CONCS'] * df['BRS'] * duration/60 * (1-df['EFFIN']) 
    df['PS_F']  = 1 - np.exp(-df['INS_F']) #<--Per class infection rate for faculty
    df['PF_S']  = 1 - np.exp(-df['INF_S'])
    df['PS_S']  = 1 - np.exp(-df['INS_S'])
    #FACULTY INFECTION PROBABILITIES
    df['nPS_F'] = 1 - df['PF_S']           #<--Faculty not infected probability per class
    df['nPS_Fsemester'] = df['nPS_F']**num_class_periods
    df['PS_Fsemester']  = 1 - df['nPS_Fsemester'] #<--Faculty infection probability per semester
    #STUDENT INFECTION PROBABILITIES
    df['PS_FS'] = df['PS_F']+df['PS_S']    #<--Per class infection rate for student
    df['nPS_FS'] = 1 - df['PS_FS']         #<--Per class non-infection rate for student  
    df['nPS'] = df['nPS_FS']**(num_classes_taken*num_class_periods)
    df['PS_Ssemester'] = 1 - df['nPS']
    return(df)

def update_figure(df,faculty=True):
    if faculty: fld = 'PS_Fsemester'; txt = 'Faculty'
    else: fld = 'PS_Ssemester'; txt = 'Student'
    #Get the max x value
    x_max = df['PS_Ssemester'].max()
    #Update the figure
    fig = px.histogram(df,x=fld,nbins=40,
                       title=f'Distribution of {txt} Infection Probabilities over<br>the course of the Semester for 10,000 Monte Carlo Simulations')
    fig.update_xaxes(title_text = 'Probability of infection (%)',
                     range=[0,x_max])
    fig.update_layout(xaxis_tickformat = "%",
                      font_size=10)
    #fig.update_layout(transition_duration=500)
    return(fig)

def summarize_output(df,faculty=True):
    if faculty: fld = 'PS_Fsemester'; txt = 'Faculty Member'
    else: fld = 'PS_Ssemester'; txt = 'Student'
    #Create markdown from values
    the_mean = df[fld].mean()
    the_quants = [df[fld].quantile(x) for x in (0.05,0.25,0.5,0.75,0.95)]
    #Create Markdown
    md_text=f'''  
**Average Infection Probability for {txt} for semester**

| Average (10k runs) | {the_mean:0.2%} |
| --: | --- |
| 5th percentile: | {the_quants[0]:0.2%} |
| 25th percentile: | {the_quants[1]:0.2%} |
| 50th percentile: | {the_quants[2]:0.2%} |
| 75th percentile: | {the_quants[3]:0.2%} |
| 95th percentile: | {the_quants[4]:0.2%} |
'''
    return md_text

#%%Read in the static data
df = update_df()
fig = update_figure(df)
md_results = summarize_output(df)

#%% Page construction
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
app.title = "COVID exposure modeler"
server = app.server 

#Construct the web site
app.layout = html.Div([
    dcc.Markdown('''
### Monte Carlo Estimation of COVID-19 airborne transmission during classroom teaching: 

This is a Monte Carlo verison of Prof. Jose Jimenez’s classroom/semester sheets 
of his COVID-19 risk estimator ([https://tinyurl.com/covid-estimator](https://tinyurl.com/covid-estimator)). 
Please see the README and FAQ tabs on his worksheet for important information 
on assumptions, methodology, and inputs. This Monte Carlo calculator is available for download as a a spreadsheet at https://tinyurl.com/yxfd23kr.

**_NOTE: 
The absolute estimates of risk are very uncertain, only expect to get the order-of-magnitude right. 
The effect of control measures (e.g. more ventilation, fewer people, shorter duration, masks vs not) are expected to be much more accurate. 
Please do not just latch on to the numbers, values such as the quanta emission rates evolve with new knowledge._**  

---     
Developed by **Prasad Kasibhatla** (Duke), with help from Prof. Jose Jimenez (U. Colorado) and Prof. Elizabeth Albright (Duke)  
Dashboard created by [**John Fay**](mailto:john.fay@duke.edu) (Duke) -- Code available at: [https://github.com/johnpfay/CovidExposure](https://github.com/johnpfay/CovidExposure)  
Please contact [Prasad Kasibhatla](mailto:psk9@duke.edu) if you have questions, comments, and suggestions. 
''',style={'border-style':'ridge',
           'padding':'0.5em',
           'background-color': 'lightblue'}),
    html.Table([
        html.Tr([
            html.Th("Known Variables"), 
            html.Th("Value"),
            html.Th("______",style={'color':'white'}),
            html.Th("Uncertain Variables"), 
            html.Th("Min"), 
            html.Th("Max")]),
        html.Tr([
            html.Td("Area of Room (sq.ft.)"), 
            html.Td(dcc.Input(id='surface',value=900,type='number')),
            html.Td(""),
            html.Td("Breathing rate - Faculty (m³/hour)"),
            html.Td(dcc.Input(id='breath_fmin',value=1.0,type='number')),
            html.Td(dcc.Input(id='breath_fmax',value=1.2,type='number'))]),
        html.Tr([
            html.Td("Height of Room(ft.)"), 
            html.Td(dcc.Input(id='height',value=10,type='number')),
            html.Td(""),
            html.Td("Breathing rate - Student (m³/hour)"),
            html.Td(dcc.Input(id='breath_smin',value=0.7,type='number')),
            html.Td(dcc.Input(id='breath_smax',value=0.9,type='number'))]),
        html.Tr([
            html.Td("# of students"), 
            html.Td(dcc.Input(id='num_students',value=10,type='number')),
            html.Td(""),
            html.Td("Ventilation w/outside air (1/hour)"),
            html.Td(dcc.Input(id='vent_min',value=2,type='number')),
            html.Td(dcc.Input(id='vent_max',value=4,type='number'))]),
        html.Tr([
            html.Td("Class duration (min.)"), 
            html.Td(dcc.Input(id='class_duration',value=75,type='number')),
            html.Td(""),
            html.Td("Decay rate of the virus (1/hour)"),
            html.Td(dcc.Input(id='decay_min',value=0,type='number')),
            html.Td(dcc.Input(id='decay_max',value=0.63,type='number')),]),
        html.Tr([
            html.Td("# of class periods"), 
            html.Td(dcc.Input(id='class_periods',value=26,type='number')),
            html.Td(""),
            html.Td("Deposition to surfaces (1/hour)"),
            html.Td(dcc.Input(id='depos_min',value=0.3,type='number')),
            html.Td(dcc.Input(id='depos_max',value=1.5,type='number')),]),
        html.Tr([
            html.Td("# of classes taken per student"), 
            html.Td(dcc.Input(id='classes_taken',value=4,type='number')),
            html.Td(""),
            html.Td("Additional control measures (1/hour)"),
            html.Td(dcc.Input(id='additional_min',value=0,type='number')),
            html.Td(dcc.Input(id='additional_max',value=0,type='number')),]),
        html.Tr([
            html.Td("# of faculty in class"), 
            html.Td("1 (fixed)",style={'border-style':'solid',
                                       'border-color':'grey',
                                       'border-width':'thin'}),
            html.Td(""),
            html.Td("Quanta emission rate - faculty (quanta/hour)"),
            html.Td(dcc.Input(id='qfac_min',value=100,type='number')),
            html.Td(dcc.Input(id='qfac_max',value=300,type='number')),]),
        html.Tr([
            html.Td(), 
            html.Td(),
            html.Td(""),
            html.Td("Quanta emission rate - student (quanta/hour)"),
            html.Td(dcc.Input(id='qstu_min',value=10,type='number')),
            html.Td(dcc.Input(id='qstu_max',value=30,type='number')),]),
        html.Tr([
            html.Td(), 
            html.Td(),
            html.Td(),
            html.Td("Exhalation mask efficiency (%)"),
            html.Td(dcc.Input(id='exmask_min',value=50,type='number')),
            html.Td(dcc.Input(id='exmask_max',value=70,type='number')),]),
        html.Tr([
            html.Td(""), 
            html.Td(""),
            html.Td(""),
            html.Td("Inhalation mask efficiency (%)"),
            html.Td(dcc.Input(id='inmask_min',value=30,type='number')),
            html.Td(dcc.Input(id='inmask_max',value=50,type='number')),]),
        html.Tr([
            html.Td(),html.Td(),html.Td(),
            html.Td("Student/faculty background infection rate (%)"),
            html.Td(dcc.Input(id='infect_min',value=0.19,type='number')),
            html.Td(dcc.Input(id='infect_max',value=0.38,type='number')),]),
            ]),
              
    html.Button(id='submit-button-state',n_clicks=0,children='Recalculate',
                style={'font-size':24,'background-color': '#4CAF50',
                       'color':'white',
                       'padding':'15px 32px'}),
    html.Table([
        html.Tr([
            html.Td(dcc.Markdown(id='faculty_results')),
            html.Td(dcc.Graph(id='faculty_histogram'))
            ]),
        html.Tr([
            html.Td(dcc.Markdown(id='student_results')),
            html.Td(dcc.Graph(id='student_histogram'))
            ])
        ])
    ])
 

@app.callback([Output('faculty_results','children'),
               Output('student_results','children'),
               Output('faculty_histogram','figure'),
               Output('student_histogram','figure')],
              [Input('submit-button-state','n_clicks')],
              [State('surface','value'),
               State('height','value'),
               State('num_students','value'),
               State('class_duration','value'),
               State('class_periods','value'),
               State('classes_taken','value'),
               State('breath_fmin','value'),State('breath_fmax','value'),
               State('breath_smin','value'),State('breath_smax','value'),
               State('vent_min','value'),State('vent_max','value'),
               State('decay_min','value'),State('decay_max','value'),
               State('depos_min','value'),State('depos_max','value'),
               State('additional_min','value'),State('additional_max','value'),
               State('qfac_min','value'),State('qfac_max','value'),
               State('qstu_min','value'),State('qstu_max','value'),
               State('exmask_min','value'),State('exmask_max','value'),
               State('inmask_min','value'),State('inmask_max','value'),
               State('infect_min','value'),State('infect_max','value')]
)
def update_page(input_value,sa,ht,nstudents,cduration,cperiods,ctaken,
                breath_fmin,breath_fmax,
                breath_smin,breath_smax,
                vent_min,vent_max,
                decay_min,decay_max,
                depos_min,depos_max,
                additional_min,additional_max,
                qfac_min,qfac_max,
                qstu_min,qstu_max,
                exmask_min,exmask_max,
                inmask_min,inmask_max,
                infect_min,infect_max):
    #Recompute the monte carlo run
    df = update_df(surface_area=sa,
                   height=ht,
                   num_students=nstudents,
                   duration=cduration,
                   num_class_periods=cperiods,
                   num_classes_taken=ctaken,
                   breathing_rate_faculty = [breath_fmin,breath_fmax],
                   breathing_rate_student = [breath_smin,breath_smax],
                   ventilation_w_outside_air = [vent_min,vent_max],
                   decay_rate_of_virus = [decay_min,decay_max],
                   deposition_to_surface = [depos_min,depos_max],
                   additional_control_measures = [additional_min,additional_max],
                   quanta_emission_rate_faculty = [qfac_min,qfac_max],
                   quanta_emission_rate_student = [qstu_min,qstu_max],
                   exhalation_mask_efficiency = [exmask_min/100,exmask_max/100],
                   inhalation_mask_efficiency = [inmask_min/100,inmask_max/100],
                   background_infection_rate = [infect_min/100,infect_max/100])
                   
    #Get summaries
    fac_results = summarize_output(df,True)
    stu_results = summarize_output(df,False)
    #Create histogram
    fac_fig = update_figure(df,True)
    stu_fig = update_figure(df,False)
    #return f'Pexp Faculty (Semester): {mean_val:0.2%}',fig
    return fac_results,stu_results,fac_fig,stu_fig
    



if __name__ == '__main__':
    app.run_server(debug=True)
