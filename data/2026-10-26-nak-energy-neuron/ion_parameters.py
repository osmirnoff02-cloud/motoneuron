"""[codex] Effective Na/K and NCX extension; concentrations uM, time s.

Stoichiometry is established; kinetics and basal-background decomposition
remain scenarios. Initial Na_i/K_o are inferred from prior effective ENa/EK.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import sys,numpy as np
from numba import njit
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'2026-10-26-connected-spatial-neuron'))
from connected_parameters import inputs,schedule,ConnectedParameters
CONVERSION=1e6/96485.33212*1000. # pA / um3 -> uM/s (one elementary charge)
@dataclass(frozen=True)
class IonParameters:
    enabled:bool=True
    temperature_C:float=23.
    sodium_out0_uM:float=145000.
    potassium_in0_uM:float=140000.
    calcium_out_uM:float=2000.
    max_pump_cycles_uM_s:float=200.
    pump_Na_half_uM:float=15000.
    pump_Na_hill:float=3.
    pump_K_half_uM:float=1500.
    pump_ATP_half_uM:float=300.
    outside_volume_ratio:float=5.
    bath_recovery_s:float=1.
    ion_diffusion_um2_s:float=100.
    axial_Na_fraction:float=.1
    ncx_capacity_uM_s:float=1.
    ncx_reversal:bool=True
    provenance:str='3Na/2K/1ATP from primary pump literature; all kinetics/geometric bath settings are hypotheses, not fitted human MN data'
    def validate(self):
        vals=[v for v in asdict(self).values() if type(v) is float]
        if not np.isfinite(vals).all() or min(vals)<0 or self.temperature_C>50 or not 0<=self.axial_Na_fraction<=1:raise ValueError('Invalid ion parameters')
        if min(self.sodium_out0_uM,self.potassium_in0_uM,self.calcium_out_uM,self.max_pump_cycles_uM_s,self.pump_Na_half_uM,self.pump_Na_hill,self.pump_K_half_uM,self.pump_ATP_half_uM,self.outside_volume_ratio,self.bath_recovery_s)<=0:raise ValueError('Positive ion/pump/bath constants required')
        if not self.provenance.strip():raise ValueError('Provenance required')
    def vector(self):
        self.validate()
        return np.array([float(self.enabled),.08617333262*(273.15+self.temperature_C),self.sodium_out0_uM,self.potassium_in0_uM,self.calcium_out_uM,
          self.max_pump_cycles_uM_s,self.pump_Na_half_uM,self.pump_Na_hill,self.pump_K_half_uM,self.pump_ATP_half_uM,
          self.outside_volume_ratio,self.bath_recovery_s,self.ion_diffusion_um2_s,self.axial_Na_fraction,float(self.ncx_reversal),self.ncx_capacity_uM_s])
ION_RANGES=dict(max_pump_cycles_uM_s=[20.,150.,200.,2000.],pump_Na_half_uM=[10000.,15000.,33000.],pump_Na_hill=[1.,3.],
 pump_K_half_uM=[500.,1500.,3000.],pump_ATP_half_uM=[50.,300.,1000.],outside_volume_ratio=[1.,5.,20.],
 bath_recovery_s=[.1,1.,10.],ion_diffusion_um2_s=[0.,100.,500.],axial_Na_fraction=[.05,.1,.2],
 sodium_out0_uM=[135000.,145000.,150000.],potassium_in0_uM=[120000.,140000.,150000.],temperature_C=[20.,23.,37.],
 calcium_out_uM=[1000.,2000.],pump_support=[0.,.25,1.],ncx_capacity_uM_s=[.1,1.,10.])
ION_NAMES=tuple(k+'_'+n for k in ('soma','process') for n in ('Na_i_uM','K_i_uM','Na_o_uM','K_o_uM','NaK_ATP_uM',
 'Na_channel_out_uM','K_channel_out_uM','Na_NCX_in_uM','Na_background_in_uM','K_background_in_uM',
 'Na_bath_added_uM','K_bath_added_uM','Na_axial_amount_uM_um3','K_axial_amount_uM_um3',
 'NaK_charge_pC','NaK_incremental_ATP_uM','Nav_inward_uM','removed_proxy_NCX_ATP_uM','AMPA_Na_out_uM','AMPA_K_out_uM'))

@njit
def pump(nai,ko,atp,v):
    n=max(nai,0.)**v[7];return v[5]*n/(v[6]**v[7]+n)*max(ko,0.)/(v[8]+max(ko,0.))*max(atp,0.)/(v[9]+max(atp,0.))

@njit
def ncx_flux(ca,nai,nao,vm,vm0,ca0,na0,reverse_ratio0,v):
    dv=max(-30.,min(30.,(vm-vm0)/(2*v[1])))
    a=(max(nao,0.)/v[2])**3*(max(ca,0.)/ca0)*np.exp(-dv)
    b=(max(nai,0.)/na0)**3*np.exp(dv)*reverse_ratio0
    return v[15]*(a-b)/(1+a+b)

@njit
def currents(z,p,c,gv,cv,fb,op,capacity,R,ep,iv):
    ina=np.zeros(2);ik=np.zeros(2);na_old=np.zeros(2);k_old=np.zeros(2);ampa_na=np.zeros(2);ampa_k=np.zeros(2);oldmono=np.zeros(2)
    cp=0.
    for n in range(5):cp+=ep[3+n]*(1-min(1.,max(0.,R)))**n
    gcp=ep[8]*cp*ep[10]*ep[11];gci=ep[8]*(1-cp)
    gca=(gci*ep[12]+gcp*ep[13])/3.;gmono=gcp+gci-gca
    emono=-gca*120/gmono if gmono>0 else 0.
    fracNa=(emono-c[4])/(c[3]-c[4]);sk0=cv[7]
    for j in range(2):
        b=185+7*j;k=199+20*j;vm=z[b];m,h,n,w=z[b+1:b+5];frac=gv[j]/(gv[0]+gv[1]);cap=c[0]*frac
        ena=iv[1]*np.log(max(z[k+2],1e-12)/max(z[k],1e-12));ek=iv[1]*np.log(max(z[k+3],1e-12)/max(z[k+1],1e-12))
        loss=1-fb[0]*min(1.,max(0.,1-z[180])) if j else 1.
        gna=cap*p[0]*loss*m**3*h;gk=cap*(p[3]*n**4+p[12]*w)+fb[3]*frac*(z[b+6]-sk0)
        ina[j]=gna*(vm-ena);ik[j]=gk*(vm-ek);na_old[j]=gna*(vm-c[3]);k_old[j]=gk*(vm-c[4])
        af=cv[6] if j else 1-cv[6];gg=gmono*af*op*capacity
        ampa_na[j]=gg*fracNa*(vm-ena);ampa_k[j]=gg*(1-fracNa)*(vm-ek);oldmono[j]=gg*(vm-emono)
    return ina,ik,na_old,k_old,ampa_na,ampa_k,oldmono

def prepare_ions(base,p,c,gv,cv,fb,ep,c0,volumes,nc0,ip,leak_fraction=None):
    iv=ip.vector();na0=iv[2]*np.exp(-c[3]/iv[1]);ko0=iv[3]*np.exp(c[4]/iv[1])
    ext=np.zeros(40)
    for j in range(2):ext[20*j:20*j+4]=[na0,iv[3],iv[2],ko0]
    b=np.r_[base,ext];cv2=np.r_[cv,c0**cv[4]/(cv[3]**cv[4]+c0**cv[4])]
    cur=currents(b,p,c,gv,cv2,fb,0.,1.,b[106],ep,iv)
    # Optional joint pump/leak calibration. It is fixed BEFORE integration and
    # never uses a target K concentration during a trajectory. Column9 is a
    # constant pump-capacity multiplier, not an ion-restoring flux.
    if leak_fraction is not None and (not np.isfinite(leak_fraction) or not 0<leak_fraction<=1):
        raise ValueError('Joint NaK leak fraction must be in (0,1]')
    baseline=np.zeros((2,9 if leak_fraction is None else 10))
    for j in range(2):
        atp=b[120 if j==0 else 136];vlocal=iv.copy();vlocal[5]*=(gv[j]/volumes[j])/(gv[0]/volumes[0])
        J=pump(na0,ko0,atp,vlocal);F=CONVERSION/volumes[j]
        reverse_ratio=(na0/iv[2])**3*(iv[4]/c0)*np.exp(gv[3]/iv[1])
        new_nc0=iv[15]*(1-reverse_ratio)/(2+reverse_ratio) if iv[14] else nc0[j]
        multiplier=1.
        if leak_fraction is not None:
            vm=b[185+7*j];ena=c[3];ek=c[4]
            if not ek<vm<ena:raise ValueError('Joint leak closure needs EK < resting V < ENa')
            cap=c[0]*gv[j]/(gv[0]+gv[1])
            gn=-(cur[0][j]*F+3*J-3*new_nc0)/F/(vm-ena)
            gk=-(cur[1][j]*F-2*J)/F/(vm-ek)
            slope=(-3/(vm-ena)+2/(vm-ek))/F
            desired=J+(leak_fraction*cap*gv[5]-gn-gk)/slope
            if desired<=0:raise ValueError('Joint leak closure requires a positive pump capacity')
            multiplier=desired/J;J=desired
        # Fixed resting backgrounds; no division by the NCX driving force near reversal.
        bn=cur[0][j]*F+3*J-3*new_nc0;bk=cur[1][j]*F-2*J
        if ip.enabled and (bn < -1e-9 or bk > 1e-9):
            raise ValueError('Resting background would require active Na export or K import without an energy owner; increase capacity or recalibrate resting channel currents')
        baseline[j,:9]=[J,cur[0][j],cur[1][j],bn,bk,nc0[j],reverse_ratio,atp,new_nc0]
        if leak_fraction is not None:baseline[j,9]=multiplier
    return b,iv,cv2,baseline

@njit
def ion_extension(z,old,row,drive,srow,injection,ctx,iv,ib,cv2,pump_support,conn_factor):
    # ctx tuple belongs to unmodified API0.11. Outgoing charges in pA.
    out=np.zeros(239);out[:199]=old
    if iv[0]==0:return out
    p,c,gv,fb,c0=ctx[36],ctx[37],ctx[38],ctx[40],ctx[41]
    # Indices are verified against named ctx in builder and tests.
    ep=ctx[9];en=ctx[14];vs,vp=ctx[27],ctx[28];sk,pk=ctx[30],ctx[32]
    cv=cv2.copy();cv[6]=srow[2]
    ina,ik,naold,kold,an,ak,om=currents(z,p,c,gv,cv,fb,row[2],row[23],z[106],ep,iv)
    vols=np.array([vs,vp]);ncidx=np.array([131,145]);atpidx=np.array([120,136]);proxyidx=np.array([128,152]);vidx=np.array([185,192])
    for j in range(2):
        b=199+20*j;at=atpidx[j];vm=z[vidx[j]];vol=vols[j];fac=CONVERSION/vol
        vlocal=iv.copy();vlocal[5]*=(gv[j]/vol)/(gv[0]/vs)
        cap=c[0]*gv[j]/(gv[0]+gv[1]);J=pump(z[b],z[b+3],z[at],vlocal)*pump_support
        if ib.shape[1]>9:J*=ib[j,9]
        oldnc=old[ncidx[j]];nc=oldnc
        if iv[14]>0:
            ca=max(c0+z[0 if j==0 else 134],0.)
            na0=iv[2]*np.exp(-c[3]/iv[1])
            nc=ncx_flux(ca,z[b],z[b+2],vm,gv[3],c0,na0,ib[j,6],iv)
        delta=nc-oldnc;out[ncidx[j]]=nc
        basal_delta=ib[j,8]-ib[j,5]
        if j==0:
            out[0]-=(delta-basal_delta)/(1+sk);out[23]+=(delta-basal_delta)/(1+sk);out[132]+=basal_delta
        else:
            out[134]-=(delta-basal_delta)/(1+pk);out[146]+=basal_delta
        proxy=old[proxyidx[j]];out[proxyidx[j]]=0.
        out[at]+=proxy-J;out[at+1]-=proxy-J
        # Replace old indirect NCX reserve, add a fixed resting NaK reserve once.
        reserve=ib[j,0]-ib[j,5]*en[10]
        ox=reserve*(1-en[8])*drive[0]*max(z[at+1],0.)/(en[6]+max(z[at+1],0.))*(en[6]+en[1])/en[1]
        gly=reserve*en[8]*drive[1]*max(z[at+1],0.)/(en[7]+max(z[at+1],0.))*(en[7]+en[1])/en[1]
        out[at]+=ox+gly;out[at+1]-=ox+gly;out[123 if j==0 else 148]+=ox;out[124 if j==0 else 149]+=gly
        ipump=(J-ib[j,0])/fac;incx=-(nc-ib[j,8])/fac
        change=ina[j]-naold[j]+ik[j]-kold[j]+an[j]+ak[j]-om[j]+ipump+incx
        out[vidx[j]]-=1000*change/cap
        fn=(ina[j]+an[j])*fac;fk=(ik[j]+ak[j])*fac
        dn=-fn-3*J+3*nc+ib[j,3];dk=-fk+2*J+ib[j,4]
        out[b]=dn;out[b+1]=dk
        bathna=(iv[2]-z[b+2])/iv[11];ko0=iv[3]*np.exp(c[4]/iv[1]);bathk=(ko0-z[b+3])/iv[11]
        out[b+2]=-dn/iv[10]+bathna;out[b+3]=-dk/iv[10]+bathk
        out[b+4]=J;out[b+5]=fn;out[b+6]=fk;out[b+7]=3*nc;out[b+8]=ib[j,3];out[b+9]=ib[j,4]
        out[b+10]=bathna*iv[10];out[b+11]=bathk*iv[10]
        out[b+14]=J/fac;out[b+15]=J-ib[j,0];out[b+16]=max(-ina[j],0.)*fac;out[b+17]=proxy;out[b+18]=an[j]*fac;out[b+19]=ak[j]*fac
    # Intracellular axial current and chemical diffusion, same geometry; bath has no axial exchange.
    ax=gv[2]*(z[192]-z[185])*CONVERSION
    dNa=iv[12]*conn_factor*(z[219]-z[199])+iv[13]*ax
    dK=iv[12]*conn_factor*(z[220]-z[200])+(1-iv[13])*ax
    for j in range(2):
        b=199+20*j;sign=1 if j==0 else -1
        out[b]+=sign*dNa/vols[j];out[b+1]+=sign*dK/vols[j]
        out[b+12]=sign*dNa;out[b+13]=sign*dK
    diff_charge=(iv[12]*conn_factor*(z[219]-z[199]+z[220]-z[200]))/CONVERSION
    out[185]+=1000*diff_charge/(gv[0]/100);out[192]-=1000*diff_charge/(gv[1]/100)
    return out
