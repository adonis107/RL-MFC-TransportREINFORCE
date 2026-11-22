"""Compare the two controlled traffic runs after their gap CSVs are generated."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    old_root=Path('outputs/traffic_cells3_T5_seed0')
    new_root=Path('outputs/traffic_cells3_T5_mixed_seed0')
    old_plan=json.loads((old_root/'experiment_plan.json').read_text())
    new_plan=json.loads((new_root/'experiment_plan.json').read_text())
    def metadata(plan):
        paths=list((Path(plan['training_results_root'])/'traffic').glob('transport_*/metadata.json'))
        if len(paths)!=1:
            raise ValueError('Expected exactly one completed Transport run per protocol')
        return json.loads(paths[0].read_text())
    old_metadata,new_metadata=metadata(old_plan),metadata(new_plan)
    old_env=dict(old_metadata['env_config']);new_env=dict(new_metadata['env_config'])
    old_family=old_env.pop('training_initial_law','dirichlet');new_family=new_env.pop('training_initial_law','dirichlet')
    old_env.pop('training_jam_probability',None);new_env.pop('training_jam_probability',None)
    assert old_env==new_env, 'A physical benchmark parameter changed'
    assert old_metadata['algorithm_config']==new_metadata['algorithm_config'], 'Estimator settings changed'
    assert old_metadata['perturbation']==new_metadata['perturbation']
    assert old_metadata['simulator_budget_estimate']==new_metadata['simulator_budget_estimate']
    old=np.genfromtxt(old_root/'optimality_gap.csv',delimiter=',',names=True)
    new=np.genfromtxt(new_root/'optimality_gap.csv',delimiter=',',names=True)
    np.testing.assert_array_equal(old['simulated_training_transitions'],new['simulated_training_transitions'])
    np.testing.assert_allclose(old['reference_validation_return'],new['reference_validation_return'],atol=1e-7,rtol=0)
    def stats(values):
        gap=values['absolute_gap']
        return dict(initial_gap=float(gap[0]),final_gap=float(gap[-1]),minimum_logged_gap=float(gap.min()),
                    minimum_gap_training_transitions=int(values['simulated_training_transitions'][gap.argmin()]),
                    final_100_mean_gap=float(gap[-100:].mean()),final_validation_return=float(values['validation_return'][-1]))
    report=dict(changed_parameter='Training initial laws only',old_training_family=old_family,new_training_family=new_family,
                mixed_jam_probability=new_metadata['env_config']['training_jam_probability'],same_estimator_settings=True,
                training_transitions=int(new['simulated_training_transitions'][-1]),
                lambda_=new_metadata['perturbation'],diffuse_only=stats(old),mixed=stats(new),
                scope='One paired training seed; lambda and allocation retained from diffuse-law calibration to isolate initial-law coverage.')
    (new_root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.6,3.8),constrained_layout=True)
    ax.plot(old['simulated_training_transitions']/1e6,old['absolute_gap'],color='#8e9aaf',lw=1.05,alpha=.85,label='Diffuse-only training')
    ax.plot(new['simulated_training_transitions']/1e6,new['absolute_gap'],color='#2a78d6',lw=1.2,label='50% jam / 50% diffuse training')
    ax.set(xlabel='Simulated training transitions (millions)',ylabel=r'$|J(\widehat\theta)-J(\theta^\star)|$',
           title='Traffic: same seed and estimator, different training laws',xlim=(0,63.6),ylim=(0,None))
    ax.legend(frameon=False,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.22)
    for suffix in ('png','pdf'):
        fig.savefig(new_root/f'initial_law_comparison.{suffix}',dpi=220)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
