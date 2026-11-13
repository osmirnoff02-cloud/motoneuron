"""[codex] Exploratory full-model screen; none are human confidence bounds."""
BASE=dict(g=.5,tau=.08,vhalf=-27.3966,leak=.5,pump=1.,water=.5,supply=1.,creatine=1.,replacement=.1)
SCENARIOS={'reference':BASE.copy(),'legacy':dict(BASE,absolute=False)}
for key,values in {'g':[0.,2.,8.],'tau':[.02,.3],'vhalf':[-40.,-20.],
                   'leak':[.25,.75],'pump':[.5,2.],'water':[.2,.8],
                   'supply':[.5,2.],'creatine':[.25,2.]}.items():
    for v in values:SCENARIOS[key+'_'+str(v).replace('-','neg').replace('.','p')]=dict(BASE,**{key:v})
SCENARIOS['water_pump_high']=dict(BASE,water=.8,pump=2.)
SCENARIOS['supply_creatine_high']=dict(BASE,supply=2.,creatine=2.)
SCENARIOS['g_fast']=dict(BASE,g=2.,tau=.02)
SCENARIOS['no_slow_replacement']=dict(BASE,replacement=0.)
SCENARIOS['weak_replacement']=dict(BASE,replacement=.025)
RANGES={'pump_capacity_multiplier':[.5,1.,2.], 'energy_schedule_supply':[.5,1.,2.],
        'slow_replacement_fraction':[0.,.025,.1]}

