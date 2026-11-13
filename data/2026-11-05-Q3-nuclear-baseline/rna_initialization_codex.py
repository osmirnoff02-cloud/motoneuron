"""[codex] Conditional steady RNA export at the declared initial resource state.

Solve the existing five-state RNA equations and shared NXF/UAP constraints.
No rate change, no hidden compensating flux and no new RNA species. Fixed ATP,
initial NCT and pre-ribosome pools are boundary conditions for this initializer;
this is NOT a stationary solution of the whole4413-state neuron.
"""
import numpy as np
from scipy.optimize import least_squares

def prepare(base,*,v,nv,shape,off,noff,rmkv,rppv,p,hv,nucv,kp,lmrref,nrt):
    if v[25]!=0 or shape[0]!=85:raise ValueError('Q3 initializer currently supports85 RNAs and zero extra nuclear decay')
    nr,np_,rp,rib,car,count,size=shape;r=nv[0];e=base[120]/(v[12]+base[120])
    source=np.r_[v[0]/hv[4],v[0]/p[23],np.log(2)/kp[4]*base[342:344]*v[0],lmrref[0],rmkv[0]*rppv[0]]
    kd=np.r_[np.zeros(5),rmkv[1]]
    if np.any(source<=0) or np.any(kd[5:]<=0):raise ValueError('Positive basal flux and RP decay required')
    original=base[off:off+size].copy();cyto=original[:rp].reshape(nr,8)[:,4].copy()
    maxD=v[10]*v[4]*e
    def build(a):
        U,N,Nc,gate,D=np.exp(a);P=v[9]*gate;L=v[6]*e/r;H=v[8]/r
        q=np.zeros((nr,8))
        # First five: only cytoplasmic loss/departure, so all stages carry J.
        q[:5,3]=source[:5]/D
        q[:5,2]=r*(q[:5,3]+source[:5]/P)
        q[:5,1]=source[:5]/(H*N)
        q[:5,0]=(source[:5]+v[7]*q[:5,1])/(L*U)
        q[:5,4]=cyto[:5]
        # RP RNAs: the same effective decay applies to all five species.
        k=kd[5:];a2=r*(1+(D+k)/P)
        a1=((P/r+k)*a2-P)/(H*N)
        a0=(v[7]+H*N+k)*a1/(L*U)
        c=D/k;amount=source[5:]/(k*(a0+a1+a2+1+c))
        q[5:,:5]=np.c_[a0,a1,a2,np.ones(80),c]*amount[:,None]
        x=original.copy();x[:rp]=q.ravel();x[car:car+3]=[Nc,N,U]
        actual_gate=nrt.common_pore(x,base[noff:noff+33],v,nv,shape)[0]
        release=D*q[:,3];cyt_bound_loss=kd*q[:,3]
        residual=np.array([
          (U+q[:,1].sum())/v[3]-1,
          (N+Nc+q[:,2:4].sum())/v[2]-1,
          (P*(Nc-N/r)-np.sum(release+cyt_bound_loss))/source.sum(),
          np.log(gate/actual_gate),np.log(D*(v[11]+q[:,3].sum())/maxD)])
        return x,q,residual
    guess=np.log([v[3]*.9,v[2]*r/(1+r)*.9,v[2]/(1+r)*.9,.7,maxD/v[11]])
    fit=least_squares(lambda a:build(a)[2],guess,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=500)
    x,q,resid=build(fit.x)
    if not fit.success or abs(resid).max()>1e-8:raise ArithmeticError('RNA initialization did not converge: '+str(resid))
    z=base.copy();z[off:off+size]=x;z[322]=q[0,0]/v[0];z[324]=q[1,0]/v[0]
    # Independently evaluate the existing transport kernel plus the unchanged
    # source/loss equations. Carrier pools may still feel other cargo later.
    d,_=nrt.chemistry(x,z[noff:noff+33],z[120],v,nv,np.ones(9),np.ones(7),shape)
    dr=d[:rp].reshape(nr,8).copy();dr[:,0]+=source
    dr[:5,4]-=source[:5];dr[5:,:5]-=kd[5:,None]*q[5:,:5]
    # Nuclear/cytoplasmic receptor release by RP RNA decay.
    cd=d[car:car+3].copy();cd[0]+=np.sum(kd*q[:,3]);cd[1]+=np.sum(kd*q[:,2]);cd[2]+=np.sum(kd*q[:,1])
    error=max(abs(dr[:,:5]).max(),abs(cd).max())
    if error>1e-9:raise ArithmeticError('Existing RNA RHS disagrees with initialized steady state: '+str(error))
    return z,dict(mode='conditional_RNA_export_stationary',max_RNA_carrier_rate_uM_s=float(error),
      constraint_residual=float(abs(resid).max()),solver_evaluations=fit.nfev,
      first5_cytoplasm_preserved=True,RP_total_max_error_uM=float(abs(q[5:,:5].sum(1)-source[5:]/kd[5:]).max()),
      nuclear_free_RNA_before=base[[322,324]].tolist(),nuclear_free_RNA_after=z[[322,324]].tolist(),
      scope='RNA export only at initial ATP/NCT/particle boundary; no full-neuron equilibrium, human fit or clamping during simulation')
