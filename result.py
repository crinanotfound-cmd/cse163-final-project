"""
Project Part 3
Caroline Shi
CSE 163
Summer 2026

Description···
"""
import pandas as pd
import matplotlib.pyplot as plt
from analysis import load_data, clean_data, DATA_PATH

def statewide_trend(data: pd.DataFrame):
    wa = clean_data(data).copy()
    weekly = wa.groupby('date')[['New cases',
        'New deaths']].sum()
    weekly['Case change'] = weekly['New cases'].diff()
    weekly['Death change'] = weekly['New deaths'].diff()

    max_weekly_change_case = weekly['Case change'].max()
    max_weekly_change_case_week = weekly['Case change'].idxmax()

    print(f'The week of max weekly case increase: {max_weekly_change_case_week}')
    print(f'The max weekly case increase value: {max_weekly_change_case}')
    
    max_weekly_change_death = weekly['Death change'].max()
    max_weekly_change_death_week = weekly['Death change'].idxmax()

    print(f'The week of max weekly death increase: {max_weekly_change_death_week}')
    print(f'The max weekly death increase value: {max_weekly_change_death}')

    