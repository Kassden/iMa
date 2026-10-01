"""Matched fixed-model controls and cross-pipeline feature transfer."""
import argparse
import hashlib
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ima.feature_discovery_specs import DiscoverySpec
from ima.feature_studies import paired_feature_report
from ima.research_executor import RecipeExecutionRequest, execute_recipe
from ima.research_specs import PipelineRecipe, V5_PORTFOLIO_VERSION


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    spec=DiscoverySpec.model_validate(json.loads(args.spec.read_text()))
    protocol=json.loads(args.protocol.read_text())
    digest=hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    comparisons=[]
    with threadpool_limits(limits=1):
        for model,parameters in (("benter_conditional_logit",{"l2":.1,"max_iter":200}),("boosted",{"max_iter":40,"max_leaf_nodes":15})):
            results=[]
            for arm,discovery in (("control",None),("challenger",spec)):
                recipe=PipelineRecipe(model={"kind":model,"parameters":parameters},calibration={"kind":"none"},blend={"kind":"none"},feature_discovery=discovery,seed=17)
                attempt=f"{model}-{arm}-{recipe.recipe_hash()}"
                request=RecipeExecutionRequest(attempt,"fixed-feature-study",0,recipe,args.dataset,args.output/"trials"/attempt,protocol,digest,portfolio_version=V5_PORTFOLIO_VERSION)
                result=execute_recipe(request)
                if result.status!="completed":
                    raise RuntimeError(result.error)
                results.append(result)
            report=paired_feature_report(results[1].artifacts["predictions"],results[0].artifacts["predictions"],"win_probability")
            report.update(model_kind=model,control_attempt=results[0].attempt_id,challenger_attempt=results[1].attempt_id,discovery_id=spec.discovery_id(),control_parameters=parameters)
            comparisons.append(report)
    path=args.output/"fixed-model-comparisons.json"
    path.write_text(json.dumps(comparisons,indent=2))
    print(json.dumps({"report":str(path),"comparisons":comparisons},indent=2))


if __name__=="__main__": main()
