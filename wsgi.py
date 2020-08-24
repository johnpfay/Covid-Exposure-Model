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
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
app.title = "COVID exposure modeler"
application = app.server 

#Construct the web site
app.layout = html.Div([
    html.Div([
    dcc.Markdown('''
### Estimation of COVID-19 infection risk from airborne transmission during classroom teaching 

This is a tool to help understand COVID-19 transmission risk to students and teachers/professors 
due to [transmission by microscopic airborne particles 
(i.e. aerosols)](https://science.sciencemag.org/content/368/6498/1422) in classroom settings.
This is not an [infectious disease dynamics model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology), 
but rather a model that predicts airborne 
[virion](https://theconversation.com/what-is-a-virus-how-do-they-spread-how-do-they-make-us-sick-133437) 
concentrations within a classroom, taking into account exhalation of 
virion-containing aerosols by infected individuals and the loss of these particles due to various processes. 
Probabilities of infection are calculated based on the
virion dose inhaled (accounting for use of masks) by uninfected people in the classroom. 

This probabilistic Monte Carlo framework was developed by 
[Prasad Kasibhatla](https://nicholas.duke.edu/people/faculty/kasibhatla), 
as an offshoot of the 
[COVID-19 risk estimator](https://tinyurl.com/covid-estimator) developed by 
[Jose Jimenez](https://www.colorado.edu/chemistry/jose-luis-jimenez).
Please see the README and FAQ tabs on his [worksheet](https://tinyurl.com/covid-estimator) 
for important information on assumptions, methodology, and inputs.
The Monte Carlo capability implemented here allows for estimates of confidence 
intervals for model predictions of infection probabilities.
'''),
dcc.Markdown('''
_**Important**: The risk calculations here are only for disease transmission 
by the airborne aerosol route, and do not account for transmission by droplets 
or from contaminated surfaces. The implicit assumption is that appropriate 
social distancing and hygiene protocols are strictly adhered to in the classroom. 
To the extent that this is not true, the risk of infection will be higher than 
predicted by these calculations. Users should also also bear in mind that the 
absolute estimates of predicted risk from this model are quite uncertain because 
of uncertainties in our knowledge of key parameters such as the exhalation rate 
of virion-containing aerosols by infected individuals and the percentage of 
infected individuals in the classroom. The model is nevertheless useful to 
explore the relative effects of control measures (e.g. more ventilation, 
fewer people, shorter duration, masks vs no masks) on COVID-19 transmission 
by aerosols in classrooms._
''',style={'color':'#cc0e0e'}),
dcc.Markdown('''
---     
Developed by [Prasad Kasibhatla](https://nicholas.duke.edu/people/faculty/kasibhatla), 
[Jose Jimenez](https://www.colorado.edu/chemistry/jose-luis-jimenez), 
[John Fay](https://nicholas.duke.edu/people/faculty/fay), 
[Elizabeth Albright](https://nicholas.duke.edu/people/faculty/albright), and 
[William Pan](https://nicholas.duke.edu/people/faculty/pan).  
Please contact [Prasad Kasibhatla](mailto:psk9@duke.edu) with questions, comments, and suggestions.  
Code availabile at: [https://github.com/johnpfay/Covid-Exposure-Model]

'''),
],style={'border-style':'ridge',
           'border-radius': '5px',
           'padding':'0.5em',
           'background-color': 'lightblue'}),
                 
dcc.Markdown('''
### INSTRUCTIONS
1. Specify values of input parameters relevant to your specific situation, focusing on cells highlighted in yellow.  
2. Cells highlighted in gray can be changed, but we recommended that you use the default values provided.
3. Some parameter names are clickable links - click on these names for further information for recommended values.
4. Click the green 'Calculate Infection Probability' button.
5. Results will update in a few seconds, and will be displayed below the 'Calculate Infection Probability' button. 
6. Change input parameter values and click the green 'Calculate Infection Probability' button again to see updated results.
7. To calculate the infection probability for multiple courses, [follow these directions](https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A112').
''',style={'border-style':'ridge',
           'border-radius': '5px',
           'padding':'0.5em',
           'background-color': 'lightgray'}),

    html.Tr([
        html.Th(),]),
    html.Tr([
        html.Th(),]),
    html.Tr([
        html.Th(),]),
    html.Tr([
        html.Th(),]),
    html.Table([
        html.Tr([
            html.Th("Known Parameters",style={'text-align':'left'}), 
            html.Th("Value",style={'text-align':'left'}),
            html.Th("______",style={'color':'white'}),
            html.Th("Uncertain Parameters: Specify Range",style={'text-align':'left'}), 
            html.Th("Minimum",style={'text-align':'left'}), 
            html.Th("Maximum",style={'text-align':'left'})]),
        html.Tr([
            html.Td("Number of faculty in the course"),
            html.Td("1 (fixed)",style={'border-style':'solid',
                                       'border-color':'grey',
                                       'border-width':'thin',
                                       'font-size':'small'}),
            html.Td(""),
            html.Td(html.Div([html.A('Percentage of faculty-age people in community who are infectious (%)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A4', target='_blank')])),
            html.Td(dcc.Input(id='infectf_min',value=0.70,type='number')),
            html.Td(dcc.Input(id='infectf_max',value=1.40,type='number')),]),
        html.Tr([
            html.Td("Number of students in the course"),
            html.Td(dcc.Input(id='num_students',value=10,type='number')),
            html.Td(""),
            html.Td(html.Div([html.A('Percentage of student-age people in community who are infectious (%)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A4', target='_blank')])),
            html.Td(dcc.Input(id='infects_min',value=0.70,type='number')),
            html.Td(dcc.Input(id='infects_max',value=1.40,type='number')),]),
        html.Tr([
            html.Td("Number of in-person class sessions in the course"),
            html.Td(dcc.Input(id='class_periods',value=26,type='number')),
            html.Td(""),
            html.Td(html.Div([html.A('Mask efficiency in reducing virus exhalation (%)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A30', target='_blank')])),
            html.Td(dcc.Input(id='exmask_min',value=40,type='number')),
            html.Td(dcc.Input(id='exmask_max',value=60,type='number')),]),
        html.Tr([
            html.Td("Duration of each in-person class session (min.)"),
            html.Td(dcc.Input(id='class_duration',value=75,type='number')),
            html.Td(""),
            html.Td(html.Div([html.A('Mask efficiency in reducing virus inhalation (%)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A44', target='_blank')])),
            html.Td(dcc.Input(id='inmask_min',value=30,type='number')),
            html.Td(dcc.Input(id='inmask_max',value=50,type='number')),]),
        html.Tr([
            html.Td("Floor area of classroom (sq. ft.)"),
            html.Td(dcc.Input(id='surface',value=900,type='number')),
            html.Td(""),
            html.Td(html.Div([html.A('Room air ventilation rate w/outside air (air changes per hour)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A51', target='_blank')])),
            html.Td(dcc.Input(id='vent_min',value=1,type='number')),
            html.Td(dcc.Input(id='vent_max',value=4,type='number'))]),
        html.Tr([
            html.Td("Height of classroom (ft.)"),
            html.Td(dcc.Input(id='height',value=10,type='number')),
            html.Td(""),
            html.Td(html.Div([html.A('Additional control measures (effective air changes per hour)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A66', target='_blank')])),
            html.Td(dcc.Input(id='additional_min',value=0,type='number')),
            html.Td(dcc.Input(id='additional_max',value=0,type='number')),]),
        html.Tr([
            html.Td(),
            html.Td(),
            html.Td(""),
            html.Td(html.Div([html.A('Decay rate of virus infectivity indoors (per hour)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A78', target='_blank')])),
            html.Td(dcc.Input(id='decay_min',value=0,type='number')),
            html.Td(dcc.Input(id='decay_max',value=1.0,type='number')),]),
        html.Tr([
            html.Td(),
            html.Td(),
            html.Td(""),
            html.Td(html.Div([html.A('Deposition rate of virus to surfaces (per hour)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A95', target='_blank')])),
            html.Td(dcc.Input(id='depos_min',value=0.3,type='number')),
            html.Td(dcc.Input(id='depos_max',value=1.5,type='number')),]),
        html.Tr([
            html.Td(), 
            html.Td(),
            html.Td(""),
            html.Td(html.Div([html.A('Inhalation rate: Faculty (m³/minute)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A101', target='_blank')])),
            html.Td(dcc.Input(id='breath_fmin',value=0.005,type='number')),
            html.Td(dcc.Input(id='breath_fmax',value=0.010,type='number'))]),
        html.Tr([
            html.Td(),
            html.Td(),
            html.Td(""),
            html.Td(html.Div([html.A('Inhalation rate: Student (m³/minute)', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A101', target='_blank')])),
            html.Td(dcc.Input(id='breath_smin',value=0.005,type='number')),
            html.Td(dcc.Input(id='breath_smax',value=0.007,type='number'))]),
        html.Tr([]),
        html.Tr([]),
        html.Tr([]),
        html.Tr([
            html.Th(), 
            html.Th(),
            html.Th("______",style={'color':'white'}),
            html.Th("FOR ADVANCED USERS ONLY",style={'text-align':'left'})]), 
        html.Tr([
            html.Th(html.Button(id='submit-button-state',n_clicks=0,children='Calculate Infection Probability',
                                style={'font-size':24,'background-color': '#4CAF50',
                                       'color':'white',
                                       'padding':'15px 32px'}),
                rowSpan=3,
                colSpan=2),
            html.Th("______",style={'color':'white'}),
            html.Th("Click links below before specifying",style={'text-align':'left'}),
            html.Th("Mean",style={'text-align':'left'}),
            html.Th("Standard Deviation",style={'text-align':'left'})]),
        html.Tr([
            html.Td(""),
            html.Td(html.Div([html.A('log10[Quanta emission rate: Faculty (quanta/hour)]', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A107', target='_blank')])),
            html.Td(dcc.Input(id='qfac_min',value=1.5,type='number')),
            html.Td(dcc.Input(id='qfac_max',value=0.71,type='number')),]),
        html.Tr([
            html.Td(),
            html.Td(html.Div([html.A('log10[Quanta emission rate: Student (quanta/hour)]', 
                                     href='https://docs.google.com/spreadsheets/d/1LS2f28meUwiy-AxGQXyd1ily9HPbh9hvYD48Qulaj6s/edit#gid=0&range=A107', target='_blank')])),
            html.Td(dcc.Input(id='qstu_min',value=0.69,type='number')),
            html.Td(dcc.Input(id='qstu_max',value=0.71,type='number')),]),
            ]),
              

    dcc.Markdown(id='results_text'),

    html.Table([
        html.Tr([
            html.Td(dcc.Markdown(id='faculty_results'),style={'border-style':'ridge','padding':'0.5em',
                       'background-color': 'lightgray'}),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(""),
            html.Td(dcc.Markdown(id='student_results'),style={'border-style':'ridge','padding':'0.5em',
            'background-color': 'lightgray'})
            ])
        ]),

    dcc.Markdown('''
    **_Since some of the input parameters are uncertain, calculations are perfomed for 10,000 plausible scenarios using random combinations
    of input parameter values. The results shown above represent statistical summaries from these 10,000 scenarios._**
    '''),

    ])

@app.callback([Output('faculty_results','children'),
               Output('student_results','children'),
               Output('results_text','children')],
              [Input('submit-button-state','n_clicks')],
              [State('surface','value'),
               State('height','value'),
               State('num_students','value'),
               State('class_duration','value'),
               State('class_periods','value'),
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
               State('infectf_min','value'),State('infectf_max','value'),
               State('infects_min','value'),State('infects_max','value')]
)
def update_page(num_clicks,sa,ht,nstudents,cduration,cperiods,
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
                infectf_min,infectf_max,
                infects_min,infects_max):
    #Recompute the monte carlo run
    df = update_df(surface_area=sa,
                   height=ht,
                   num_students=nstudents,
                   duration=cduration,
                   num_class_periods=cperiods,
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
                   background_infection_rate_faculty = [infectf_min/100,infectf_max/100],
                   background_infection_rate_student = [infects_min/100,infects_max/100])
                   
    #Get summaries
    # if num_clicks < 1:
        # fac_results = summarize_outputx(df,True)
        # stu_results = summarize_outputx(df,False)
    # else:
    fac_results = summarize_output(df,True)
    stu_results = summarize_output(df,False)
    #Create histogram
    # fac_fig = update_figure(df,True)
    # stu_fig = update_figure(df,False)
    #return f'Pexp Faculty (Semester): {mean_val:0.2%}',fig
    # return fac_results,stu_results,fac_fig,stu_fig
    if num_clicks < 1:
        results_md = update_results(True)
        fac_results = '',
        stu_results = ''
    else:
        results_md = update_results(False)
    return fac_results,stu_results, results_md
    
if __name__ == '__main__':
    app.run_server(debug=True)
