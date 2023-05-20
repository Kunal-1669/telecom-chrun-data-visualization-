#%%
import pandas as pd
df1=pd.read_csv("cleandata.csv")
# %%
df1.isna().sum()
# %%
import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import numpy as np
import plotly.express as px
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
from scipy.fft import fft
# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__)
server = app.server


#%%
tab1_layout=html.Div([
    html.H1("Churn by Income Group"),
    dcc.Graph(
        id="churn_by_income",
        figure={
            "data": [
                {
                    "x": df1[df1["Churn"] == "Yes"]["IncomeGroup"],
                    "y": df1[df1["Churn"] == "Yes"]["MonthlyRevenue"],
                    "type": "bar",
                    "name": "Churn = Yes"
                },
                {
                    "x": df1[df1["Churn"] == "No"]["IncomeGroup"],
                    "y": df1[df1["Churn"] == "No"]["MonthlyRevenue"],
                    "type": "bar",
                    "name": "Churn = No"
                }
            ],
            "layout": {
                "title": "Monthly Revenue by Income Group and Churn",
                "xaxis": {"title": "Income Group"},
                "yaxis": {"title": "Monthly Revenue"}
            }
        }
    ),
    html.Div(
        id="avg_metrics",
        children=[
            html.H2("Average Monthly Revenue: ${:.2f}".format(df1["MonthlyRevenue"].mean())),
            html.H2("Average Months in Service: {:.2f}".format(df1["MonthsInService"].mean())),
            html.H2("Average Monthly Minutes: {:.2f}".format(df1["MonthlyMinutes"].mean()))
        ]
    )
])

tab2_layout = html.Div([
    html.H1("Customer Segmentation"),
    html.Label("Churn Status"),
    dcc.RadioItems(
        id="churn-status",
        options=[
            {"label": "Churned", "value": "Yes"},
            {"label": "Retained", "value": "No"}
        ],
        value="Yes",
        labelStyle={"display": "inline-block"}
    ),
    html.Br(),
    html.Label("Independent Variable"),
    dcc.Dropdown(
        id="independent-variable",
        options=[
            {"label": "Marital Status", "value": "MaritalStatus"},
            {"label": "Occupation", "value": "Occupation"},
            {"label": "Credit Rating", "value": "CreditRating"},
            {"label": "PRIZM Code", "value": "PrizmCode"},
            {"label": "AgeHH1", "value": "Age"}
        ],
        value="MaritalStatus"
    ),
    dcc.Graph(
        id="segmentation-chart",
        figure=go.Figure()
    )
])

@app.callback(
    Output("segmentation-chart", "figure"),
    Input("churn-status", "value"),
    Input("independent-variable", "value")
)
def update_segmentation_chart(churn_status, independent_variable):
    filtered_df = df1[df1["Churn"] == churn_status]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=filtered_df[independent_variable].value_counts().index,
            y=filtered_df[independent_variable].value_counts().values,
            name=independent_variable
        )
    )
    
    fig.update_layout(
        title="Customer Segmentation by {} and {}".format(churn_status, independent_variable),
        height=600,
        width=1000,
        showlegend=False
    )
    
    return fig

dropdown_options = [
    {'label': 'Monthly Revenue', 'value': 'MonthlyRevenue'},
    {'label': 'Monthly Minutes', 'value': 'MonthlyMinutes'},
    {'label': 'Total Recurring Charge', 'value': 'TotalRecurringCharge'}
]

tab3_layout= html.Div([
    dcc.Tabs([
        dcc.Tab(label='Line Graphs', children=[
            html.Div([
                html.Label('Select a variable'),
                dcc.Dropdown(
                    id='dropdown',
                    options=dropdown_options,
                    value='MonthlyRevenue'
                ),
                dcc.Graph(id='line-graph')
            ])
        ])
    ])
])

@app.callback(
    dash.dependencies.Output('line-graph', 'figure'),
    [dash.dependencies.Input('dropdown', 'value')]
)
def update_line_graph(selected_value):
    fig = px.box(df1, x='Churn', y=selected_value, color='Churn')
    return fig


tab4_layout= html.Div([
    

    html.H1("Scatter Plot Dashboard"),
    

    dcc.RadioItems(
        id='x-axis',
        options=[
            {'label': 'MonthlyRevenue', 'value': 'MonthlyRevenue'},
            {'label': 'MonthlyMinutes', 'value': 'MonthlyMinutes'},
            {'label': 'TotalRecurringCharge', 'value': 'TotalRecurringCharge'}
        ],
        value='MonthlyRevenue'
    ),

    dcc.RadioItems(
        id='y-axis',
        options=[
            {'label': 'MonthlyRevenue', 'value': 'MonthlyRevenue'},
            {'label': 'MonthlyMinutes', 'value': 'MonthlyMinutes'},
            {'label': 'TotalRecurringCharge', 'value': 'TotalRecurringCharge'}
        ],
        value='MonthlyMinutes'
    ),
    

    html.Div(id='scatter-plot'),
])


@app.callback(
    dash.dependencies.Output('scatter-plot', 'children'),
    [dash.dependencies.Input('x-axis', 'value'),
     dash.dependencies.Input('y-axis', 'value')])
def update_scatter_plot(x_axis_column, y_axis_column):
    fig1 = px.scatter(df1, x=x_axis_column, y=y_axis_column, color='Churn')
    return dcc.Graph(figure=fig1)


cat_cols = ['Churn', 'ServiceArea', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 
            'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel', 
            'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle', 
            'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus']
tab5_layout=html.Div([
    html.H1("Categorical Data Pie Chart"),
    dcc.Dropdown(
        id='cat-dropdown',
        options=[{'label': col, 'value': col} for col in cat_cols],
        value=cat_cols[0]
    ),
    dcc.Graph(id='pie-chart')
])


@app.callback(
    dash.dependencies.Output('pie-chart', 'figure'),
    [dash.dependencies.Input('cat-dropdown', 'value')]
)
def update_pie_chart(cat_col):
    fig = px.pie(df1, names=cat_col)
    return fig


tab6_layout= html.Div(children=[
    

    html.H1(children='KDE Plot with respect to Churn'),
    
    html.Label('Select Variable:'),
    dcc.Dropdown(
        id='dropdown1',
        options=[{'label': col, 'value': col} for col in df1.select_dtypes(include='number').columns],
        value='MonthlyRevenue'
    ),
    
    dcc.Graph(
        id='kde-plot',
        figure={}
    ),
    

    html.Label('Select Churn Status:'),
    dcc.RadioItems(
        id='radio1',
        options=[{'label': 'Churn', 'value': 'Yes'},
                 {'label': 'Non-Churn', 'value': 'No'}],
        value='Yes'
    ),
    
])


@app.callback(
    dash.dependencies.Output('kde-plot', 'figure'),
    [dash.dependencies.Input('dropdown1', 'value'),
     dash.dependencies.Input('radio1', 'value')]
)
def update_kde_plot(selected_variable, churn_status):

    df_filtered = df1[df1['Churn']==churn_status]

    fig = px.histogram(df_filtered, x=selected_variable, color='Churn', nbins=30, barmode='overlay', histnorm='probability density')
    fig.update_layout(title=selected_variable + ' KDE Plot (' + churn_status + ' Customers)')
    return fig



cols2 = df1.columns[2:]


tab7_layout= html.Div(children=[
    

    html.H1(children='Violin Plot with respect to Churn'),
    

    html.Label('Select Variable:'),
    dcc.Dropdown(
        id='dropdown1',
        options=[{'label': col, 'value': col} for col in df1.select_dtypes(include='number').columns],
        value='MonthlyRevenue'
    ),
    

    dcc.Graph(
        id='violin-plot',
        figure={}
    ),
    

    html.Label('Select Churn Status:'),
    dcc.RadioItems(
        id='radio1',
        options=[{'label': 'Churn', 'value': 'Yes'},
                 {'label': 'Non-Churn', 'value': 'No'}],
        value='Yes'
    ),
    
])


@app.callback(
    dash.dependencies.Output('violin-plot', 'figure'),
    [dash.dependencies.Input('dropdown1', 'value'),
     dash.dependencies.Input('radio1', 'value')]
)
def update_violin_plot(selected_variable, churn_status):
    # Filter the data by churn status
    df_filtered = df1[df1['Churn']==churn_status]
    # Create the violin plot
    fig = px.violin(df_filtered, x='Churn', y=selected_variable, box=True, points='all', color='Churn')
    fig.update_layout(title=selected_variable + ' Violin Plot (' + churn_status + ' Customers)')
    return fig

tab8_layout=html.Div([
    
    html.H1("Churn vs Income Group"),
    
    html.Div([
        dcc.Input(id="income-input", type="number", placeholder="Enter Income Group..."),
        html.Button("Submit", id="income-submit"),
    ]),
    
    html.Div([
        dcc.Checklist(
            id="churn-checkbox",
            options=[{"label": "Churn", "value": "Yes"},
                     {"label": "Retain", "value": "No"}],
            value=[],
        ),
    ]),
    
    dcc.Graph(id="churn-income-plot"),
])



@app.callback(
    dash.dependencies.Output("churn-income-plot", "figure"),
    [dash.dependencies.Input("income-submit", "n_clicks")],
    [dash.dependencies.State("income-input", "value"),
     dash.dependencies.State("churn-checkbox", "value")]
)
def update_graph(n_clicks, income_input, churn_value):
    if n_clicks is None:
        return dash.no_update
    
    filtered_df = df1[df1["IncomeGroup"] == income_input]
    if "Yes" in churn_value:
        filtered_df = filtered_df[filtered_df["Churn"] == "Yes"]
    elif 'No' in churn_value:
        filtered_df = filtered_df[filtered_df["Churn"] == "Yes"]
    
    fig = px.histogram(filtered_df, x="MonthlyRevenue")
    
    return fig

tab9_layout= html.Div([
    html.H1('Monthly Revenue vs Monthly Minutes'),
    html.Div([
        html.Label('Overage Minutes Range:'),
        dcc.RangeSlider(
    id='overage-slider',
    min=0,
    max=100,
    step=1,
    value=[0, 100],
    marks={i: str(i) for i in range(0, 101, 20)}
),
        html.Br(),
        html.Label('Churn:'),
        dcc.Checklist(
            id='churn-checkbox',
            options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}],
            value=['Yes', 'No']
        )
    ], style={'width': '50%', 'margin': 'auto'}),
    dcc.Graph(id='revenue-vs-minutes-graph')
])


@app.callback(
    dash.dependencies.Output('revenue-vs-minutes-graph', 'figure'),
    
    dash.dependencies.Input('overage-slider', 'value'),
    dash.dependencies.Input('churn-checkbox', 'value')
)
def update_figure(overage_range, churn):
    # Filter the data
    filtered_df = df1[(df1['OverageMinutes'] >= overage_range[0]) & (df1['OverageMinutes'] <= overage_range[1]) & (df1['Churn'].isin(churn))]
    
    # Calculate the average revenue by income group
    avg_revenue = filtered_df.groupby('IncomeGroup')['MonthlyRevenue'].mean().reset_index()
    
    # Create the scatter plot
    # fig = px.scatter(filtered_df, x='MonthlyMinutes', y='MonthlyRevenue', color='Churn', 
    #                  hover_data=['IncomeGroup'], trendline='ols')
    fig = px.scatter(filtered_df, x='MonthlyMinutes', y='MonthlyRevenue', color='Churn', 
                hover_data=['IncomeGroup'])

    # Add a trendline to the scatter plot
    # fig.update_traces(trendline='lowess')
    # Add the average revenue by income group as a text box
    # for row in avg_revenue.itertuples():
    #     fig.add_annotation(x=row.MonthlyMinutes, y=row.MonthlyRevenue, 
    #                        text=f"Average Revenue: {row.MonthlyRevenue:.2f}", showarrow=False)
    
    return fig

app.layout = html.Div([
    html.H1('Telecom Churn Analysis  ', style={'textAlign': 'center'}),
    html.Br(),
    dcc.Tabs(id='hw-questions', children=[
        dcc.Tab(label='Dataset metric ', value='q1', 
                # children=[tab1_layout]
                ),
        dcc.Tab(label='Custmoer Segmentation ', value='q2',
                # children=[tab2_layout]
                ),
        dcc.Tab(label='Box Plot ', value='q3'),
        dcc.Tab(label='Scatter plot  ', value='q4'),
        dcc.Tab(label='pie chart  ', value='q5'),
        dcc.Tab(label='Kde plot ', value='q6'),
        dcc.Tab(label='Viloin plot ', value='q7'),
        dcc.Tab(label='Churn Vs income group ', value='q8'),

        dcc.Tab(label='Overage Minutes  ', value='q9')
    ]),
    
    html.Div(id='layout')
],style={"background-color": "#ffffcc"})
@app.callback(Output(component_id='layout',component_property='children'),

             Input(component_id='hw-questions',component_property='value'),

             )
def update_layout(ques):
    if ques == 'q1':
        return tab1_layout
    elif ques=='q2':
        return tab2_layout
    elif ques=='q3':
        return tab3_layout
    elif ques=='q4':
        return tab4_layout
    elif ques=='q5':
        return tab5_layout
    elif ques=='q6':
        return tab6_layout
    elif ques=='q7':
        return tab7_layout
    elif ques=='q8':
        return tab8_layout
    elif ques=='q9':
        return tab9_layout
    
app.run_server(
        port=8054,
        host='0.0.0.0'
    )
# %%
# %%



# %%
