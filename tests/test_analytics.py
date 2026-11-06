import pandas as pd
from src.analytics import portfolio_kpis, segment_profile

def test_portfolio_kpis():
    d=pd.DataFrame({'account_id':['a','b'],'target_default':[1,0],'loan_amount':[100,200],'repayment_latest_outstanding':[80,150]})
    k=portfolio_kpis(d)
    assert k['accounts']==2 and k['defaults']==1 and k['default_rate_pct']==50

def test_segment_profile():
    d=pd.DataFrame({'account_id':['a','b'],'target_default':[1,0],'loan_amount':[100,200],'repayment_latest_outstanding':[80,150],'region':['N','N']})
    assert segment_profile(d,'region').iloc[0]['accounts']==2
