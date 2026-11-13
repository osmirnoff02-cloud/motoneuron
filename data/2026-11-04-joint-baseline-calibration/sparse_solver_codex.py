"""[codex] Complete finite-difference Jacobian, stored sparsely.

No guessed sparsity pattern, threshold truncation, fixed-state elimination or
equation reduction. Recomputed at each requested state; zero columns are still
evaluated. Numerical checks against LSODA are required for each new use.
"""
import numpy as np
from scipy.sparse import csc_matrix

def make_jacobian(fun,atol,rtol,stats):
    def jac(t,y):
        f=fun(t,y);n=len(y);vals=[];indices=[];ptr=[0]
        step=np.sqrt(np.finfo(float).eps)*np.maximum(np.abs(y),np.asarray(atol)/rtol)
        for j in range(n):
            p=y.copy();p[j]+=step[j];h=p[j]-y[j]
            if h==0:raise ArithmeticError('Finite difference increment vanished')
            v=(fun(t,p)-f)/h
            k=np.flatnonzero(v) # retain every nonzero, however small
            indices.extend(k);vals.extend(v[k]);ptr.append(len(vals))
        stats['calls']+=1;stats['rhs_evaluations']+=n+1;stats['last_nnz']=len(vals)
        return csc_matrix((vals,indices,ptr),shape=(n,n))
    return jac

def make_colored_jacobian(fun,atol,rtol,stats):
    """Adaptive finite-difference approximation with checked column coloring.

    A dense-in-columns probe discovers edges at both the current state and a
    positive perturbation. The union only grows. Two independent directional
    checks on every Jacobian can force a full rebuild. This approximation never
    changes the RHS; compare trajectories/events with the exact-column mode.
    """
    full=make_jacobian(fun,atol,rtol,stats);pattern=None;groups=None
    rng=np.random.default_rng(20261104)
    stats.update(strategy='adaptive_checked_coloring',direction_checks=0,rebuilds=0,groups=0)
    def rebuild(t,y):
        nonlocal pattern,groups
        a=full(t,y);probe=y+.05*np.maximum(abs(y),np.asarray(atol)/rtol)
        b=full(t,probe)
        discovered=(abs(a)+abs(b)).astype(bool)
        pattern=discovered if pattern is None else (pattern+discovered).astype(bool)
        rows=[set(pattern.indices[pattern.indptr[j]:pattern.indptr[j+1]]) for j in range(len(y))]
        groups=[];occupied=[]
        for j in sorted(range(len(y)),key=lambda k:-len(rows[k])):
            if not rows[j]:continue
            for g,used in enumerate(occupied):
                if rows[j].isdisjoint(used):groups[g].append(j);used.update(rows[j]);break
            else:groups.append([j]);occupied.append(set(rows[j]))
        stats['rebuilds']+=1;stats['groups']=len(groups)
        return a
    def jac(t,y):
        if pattern is None:return rebuild(t,y)
        f=fun(t,y);h=np.sqrt(np.finfo(float).eps)*np.maximum(abs(y),np.asarray(atol)/rtol)
        h=(y+h)-y
        if np.any(h==0):raise ArithmeticError('Finite difference increment vanished')
        values=np.zeros(pattern.nnz)
        for group in groups:
            q=y.copy();q[group]+=h[group];df=fun(t,q)-f
            for j in group:
                sl=slice(pattern.indptr[j],pattern.indptr[j+1]);values[sl]=df[pattern.indices[sl]]/h[j]
        a=csc_matrix((values,pattern.indices,pattern.indptr),shape=pattern.shape)
        stats['calls']+=1;stats['rhs_evaluations']+=len(groups)+1
        for _ in range(2):
            v=h*rng.choice([-1.,1.],size=len(y));actual=fun(t,y+v)-f;pred=a@v
            scale=np.maximum(abs(actual),abs(pred))+1e-8*np.maximum(abs(f),1e-10)
            stats['direction_checks']+=1;stats['rhs_evaluations']+=1
            if np.max(abs(actual-pred)/scale)>0.03:return rebuild(t,y)
        stats['last_nnz']=a.nnz
        return a
    return jac
