"""Visual P0 audit after prediction inventory. No training or source mutation.

Requires an existing output from exp025_p0_audit.py, exact source ZIPs, and the
verified seed-42 best.weights.h5. Writes only inside that analysis output.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EXPECTED={
 "split":"5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49",
 "holdout":"8d5c3758b5869c3ae19238fa8d78e53ad15bb40544f4c70f632602896d80045f",
 "checkpoint":"fddcde3988af72b4446468a813a0d516f45663459fd7c761bec4aed58ff662c5",
}
CLASSES=("anthracnose","healthy","leaf_miner","red_rust","not_cashew_leaf")
SEEDS=(42,123,2026,3407,7777)

def sha_file(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def read_csv(path):
 with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def write_csv(path,rows,fields):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def zip_index(z):
 d={}
 for n in z.namelist():
  if n.lower().endswith((".jpg",".jpeg",".png")):
   bits=n.replace("\\","/").split("/")
   split_part=next((x for x in ("train","val","test") if x in bits),None)
   if split_part:
    key="/".join(bits[bits.index(split_part):])
   elif len(bits)>=2:key="/".join(bits[-2:])
   else:continue
   if key in d:raise ValueError(f"Duplicate ZIP image path {key}")
   d[key]=n
 return d

def raw_image(z,index,relative):
 if relative not in index:raise KeyError(relative)
 data=z.read(index[relative]);return data,hashlib.sha256(data).hexdigest()

def model_input(data):
 x=tf.io.decode_image(data,channels=3,expand_animations=False)
 x.set_shape([None,None,3])
 return tf.image.resize(x,[256,256],method="bilinear")

def classifier():
 inputs=keras.Input((256,256,3),name="raw_rgb")
 aug=keras.Sequential([
  layers.RandomFlip("horizontal",seed=42),layers.RandomRotation(.08,seed=42),
  layers.RandomZoom(.10,seed=42),layers.RandomContrast(.10,seed=42),
  layers.RandomBrightness(factor=.08,value_range=(0.,255.),seed=1142),
 ],name="train_only_augmentation_seed_42")(inputs)
 x=layers.Rescaling(1./255.,name="densenet_rescale")(aug)
 x=layers.Normalization(mean=[.485,.456,.406],variance=[.229**2,.224**2,.225**2],axis=-1,name="densenet_preprocess")(x)
 backbone=keras.applications.DenseNet121(include_top=False,weights=None,input_shape=(256,256,3),pooling="avg")
 backbone.trainable=False
 x=backbone(x,training=False)
 x=layers.Dropout(.4,seed=42)(x)
 output=layers.Dense(5,activation="softmax",dtype="float32",name="disease_output")(x)
 return keras.Model(inputs,output),backbone

def features(data):
 rgb=np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
 hsv=np.asarray(Image.fromarray(rgb).convert("HSV"),dtype=np.float32)
 gray=np.asarray(Image.fromarray(rgb).convert("L"),dtype=np.float32)
 try:
  from scipy.ndimage import laplace
  blur=float(np.var(laplace(gray)))
 except ImportError:blur=float("nan")
 v=hsv[:,:,2];s=hsv[:,:,1]
 return {"width":rgb.shape[1],"height":rgb.shape[0],"v_mean":float(v.mean()),
         "v_median":float(np.median(v)),"v_p10":float(np.percentile(v,10)),"v_p90":float(np.percentile(v,90)),
         "s_mean":float(s.mean()),"s_median":float(np.median(s)),
         "gray_std":float(gray.std()),"laplacian_variance":blur,
         "dark_fraction_v_lt_32":float((v<32).mean()),"bright_fraction_v_gt_245":float((v>245).mean())}

def montage(items,path,title,blinded=False):
 if not items:return
 size=240;caption=34;cols=5;rows=math.ceil(len(items)/cols)
 canvas=Image.new("RGB",(cols*size,rows*(size+caption)+44),"white")
 draw=ImageDraw.Draw(canvas);draw.text((8,8),title,fill="black")
 for i,(name,data) in enumerate(items):
  im=Image.open(io.BytesIO(data)).convert("RGB");im.thumbnail((size-12,size-12))
  x=(i%cols)*size+(size-im.width)//2;y=(i//cols)*(size+caption)+44
  canvas.paste(im,(x,y));draw.text(((i%cols)*size+5,y+size),str(i+1) if blinded else name[:34],fill="black")
 path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ("output","split_zip","holdout_zip","checkpoint"):
  ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
 args=ap.parse_args();out=args.output.resolve()
 for p in (args.split_zip,args.holdout_zip,args.checkpoint):
  if not p.is_file():raise FileNotFoundError(p)
  if out==p.resolve() or out in p.resolve().parents:raise ValueError("Source inside analysis output")
 for key,p in (("split",args.split_zip),("holdout",args.holdout_zip),("checkpoint",args.checkpoint)):
  actual=sha_file(p)
  if actual!=EXPECTED[key]:raise ValueError(f"{key} SHA mismatch: {actual}")
 inventory=read_csv(out/"tables/error_inventory_all_seeds.csv")
 priority=read_csv(out/"tables/p0_priority_cases.csv")
 if len(inventory)!=1567 or len(priority)!=27:raise ValueError("Prediction inventory incomplete")
 by_key={(r["split"],r["relative_path"]):r for r in inventory}
 if len(by_key)!=len(inventory):raise ValueError("Duplicate prediction keys")
 figures=out/"figures";cam_dir=figures/"gradcam";cam_dir.mkdir(parents=True,exist_ok=True)
 image_data={};quality=[];missing=[]
 with zipfile.ZipFile(args.split_zip) as split_z,zipfile.ZipFile(args.holdout_zip) as hold_z:
  indices={"validation":zip_index(split_z),"holdout":zip_index(hold_z)}
  archives={"validation":split_z,"holdout":hold_z}
  for row in priority:
   split,path=row["split"],row["relative_path"]
   try:data,sha=raw_image(archives[split],indices[split],path)
   except KeyError:missing.append({"asset":f"{split}:{path}","reason":"IMAGE_NOT_IN_VERIFIED_ZIP"});continue
   image_data[(split,path)]=(data,sha)
   q={"split":split,"relative_path":path,"image_sha256":sha,"priority_group":row["priority_group"],**features(data),"status":"OK"}
   quality.append(q)
  for n in (30,31,32):
   path=f"healthy/real_healthy_{n:03}.jpg"
   data,sha=raw_image(hold_z,indices["holdout"],path)
   image_data[("holdout",path)]=(data,sha)
   quality.append({"split":"holdout","relative_path":path,"image_sha256":sha,
                   "priority_group":"healthy_borderline",**features(data),"status":"OK"})
  if missing:
   write_csv(out/"tables/missing_assets.csv",missing,["asset","reason"])
   raise RuntimeError(f"Missing {len(missing)} P0 images; no CAM produced")
  write_csv(out/"tables/image_quality_features.csv",quality,list(quality[0]))
  groups=defaultdict(list)
  for item in quality:groups[item["priority_group"]].append(item)
  summary=[]
  for group,rs in groups.items():
   for metric in ("v_mean","s_mean","gray_std","laplacian_variance"):
    vals=[float(r[metric]) for r in rs]
    summary.append({"group":group,"n":len(rs),"metric":metric,"mean":float(np.mean(vals)),
                    "median":float(np.median(vals)),"status":"EXPLORATORY_SMALL_N"})
  write_csv(out/"tables/group_comparison_summary.csv",summary,list(summary[0]))
  review=read_csv(out/"tables/review_blinded_template.csv")
  blind_items=[(r["blind_id"],image_data[(r["split"],r["relative_path"])][0]) for r in review]
  montage(blind_items,figures/"healthy_originals_blinded.png","Blinded original review; IDs map to review form",blinded=False)
  holdwrong=[(r["relative_path"],image_data[(r["split"],r["relative_path"])][0]) for r in priority if r["priority_group"]=="healthy_wrong"]
  holdright=[(r["relative_path"],image_data[(r["split"],r["relative_path"])][0]) for r in priority if r["priority_group"]=="healthy_control"]
  disease=[(r["relative_path"],image_data[(r["split"],r["relative_path"])][0]) for r in priority if r["priority_group"]=="disease_to_healthy"]
  montage(holdright+holdwrong,figures/"healthy_correct_vs_wrong.png","Healthy controls then errors")
  montage(disease,figures/"disease_to_healthy_examples.png","Disease to Healthy priority cases")
  # Smoke includes five Holdout classes where available; no labels are recomputed.
  smoke=[];seen=set()
  for row in inventory:
   if row["split"]!="holdout" or row["true_label"] in seen:continue
   if row["relative_path"] not in indices["holdout"]:continue
   data,_=raw_image(hold_z,indices["holdout"],row["relative_path"])
   smoke.append((row,data));seen.add(row["true_label"])
   if len(smoke)>=4:break
  if len(smoke)<4:raise ValueError("Smoke lacks four Holdout classes")
  model,backbone=classifier();model.load_weights(args.checkpoint)
  unpooled=keras.applications.DenseNet121(include_top=False,weights=None,input_shape=(256,256,3),pooling=None)
  unpooled.set_weights(backbone.get_weights())
  W,b=model.get_layer("disease_output").get_weights()
  W=tf.constant(W,dtype=tf.float32);b=tf.constant(b,dtype=tf.float32)
  parity=[]
  def cam_forward(x):
   pre=(x/255.-tf.constant([.485,.456,.406]))/tf.constant([.229,.224,.225])
   fmap=unpooled(pre,training=False)
   pooled=tf.reduce_mean(fmap,axis=[1,2])
   logits=tf.linalg.matmul(pooled,W)+b
   return fmap,logits,tf.nn.softmax(logits)
  for row,data in smoke:
   x=model_input(data)[None,...]
   orig=model(x,training=False).numpy()[0]
   fmap,logits,rebuilt=cam_forward(x)
   rebuilt=rebuilt.numpy()[0]
   delta=float(np.max(np.abs(orig-rebuilt)))
   expected_label=row["seed42_predicted_label"]
   expected_conf=float(row["seed42_confidence"])
   pred=CLASSES[int(np.argmax(orig))];confidence=float(np.max(orig))
   pdelta=abs(confidence-expected_conf)
   parity.append({"relative_path":row["relative_path"],"true_label":row["true_label"],
                  "logged_prediction":expected_label,"model_prediction":pred,
                  "logged_confidence":expected_conf,"model_confidence":confidence,
                  "log_confidence_abs_diff":pdelta,"unpooled_max_abs_diff":delta})
   if pred!=expected_label or pdelta>1e-4 or delta>1e-5:
    write_csv(out/"tables/parity_smoke.csv",parity,list(parity[0]))
    status=json.loads((out/"execution_status.json").read_text(encoding="utf-8"))
    status.update({"images":"COMPLETE","gradcam":"BLOCKED_PARITY_FAILURE",
                   "quality_images_read":len(quality),"parity_failure":parity[-1]})
    (out/"execution_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    with (out/"REPORT_P0_HEALTHY_ERROR_ANALYSIS.md").open("a",encoding="utf-8") as f:
     f.write(f"\n\n## Visual audit and parity gate\n\nVerified ZIP hashes and measured {len(quality)} original images. "
             f"The first classifier parity check failed: {row['relative_path']}, confidence absolute difference {pdelta:.8f}; "
             f"unpooled branch difference {delta:.8g}. No CAM was exported. "
             "See `tables/parity_smoke.csv` and `figures/healthy_originals_blinded.png`.\n")
    checks=[f"{sha_file(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file() and p.name!="CHECKSUMS.sha256"]
    (out/"CHECKSUMS.sha256").write_text("\n".join(checks)+"\n",encoding="utf-8")
    raise RuntimeError(f"Inference parity failed on {row['relative_path']}; no CAM produced")
  write_csv(out/"tables/parity_smoke.csv",parity,list(parity[0]))
  cam_rows=[]
  for row in priority:
   split,path=row["split"],row["relative_path"]
   data,sha=image_data[(split,path)]
   x=model_input(data)[None,...]
   orig=model(x,training=False).numpy()[0]
   pred=CLASSES[int(np.argmax(orig))]
   if pred!=row["seed42_predicted_label"]:
    raise RuntimeError(f"Prediction mismatch on {split}:{path}; stop before bulk CAM")
   targets=[pred] if pred==row["true_label"] else [pred,row["true_label"]]
   raw=np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((256,256)))
   fig,axes=plt.subplots(1,1+len(targets),figsize=(4*(1+len(targets)),4.5))
   axes=np.atleast_1d(axes);axes[0].imshow(raw);axes[0].set_title("Original");axes[0].axis("off")
   for j,target in enumerate(targets,1):
    with tf.GradientTape() as tape:
     fmap,logits,_=cam_forward(x)
     tape.watch(fmap)
     score=logits[:,CLASSES.index(target)]
    grad=tape.gradient(score,fmap)
    if grad is None:raise RuntimeError("CAM gradient disconnected")
    alpha=tf.reduce_mean(grad,axis=[1,2],keepdims=True)
    heat=tf.nn.relu(tf.reduce_sum(alpha*fmap,axis=-1))[0]
    heat=tf.image.resize(heat[...,None],[256,256],method="bilinear")[...,0].numpy()
    maximum=float(np.max(heat))
    status="OK" if maximum>1e-12 else "DEGENERATE_CAM"
    heat=heat/maximum if maximum>1e-12 else np.zeros_like(heat)
    if heat.shape!=(256,256) or not np.isfinite(heat).all() or heat.min()<0 or heat.max()>1+1e-6:
     raise RuntimeError("Invalid CAM map")
    rel=Path("gradcam")/split/(path.replace("/","__")+f"__{target}.png")
    target_path=figures/rel;target_path.parent.mkdir(parents=True,exist_ok=True)
    plt.imsave(target_path,heat,cmap="jet",vmin=0,vmax=1)
    axes[j].imshow(raw);axes[j].imshow(heat,cmap="jet",alpha=.4,vmin=0,vmax=1)
    axes[j].set_title(f"CAM: {target}");axes[j].axis("off")
    cam_rows.append({"split":split,"relative_path":path,"image_sha256":sha,"seed":42,
     "true_label":row["true_label"],"predicted_label":pred,"confidence":row["seed42_confidence"],
     "wrong_n_seeds":row["wrong_n_seeds"],"predicted_healthy_n_seeds":row["predicted_healthy_n_seeds"],
     "model_sha256":EXPECTED["checkpoint"],"source_archive_sha256":EXPECTED["holdout" if split=="holdout" else "split"],
     "cam_target":target,"cam_layer_or_source":"unpooled DenseNet121 final 4D feature map",
     "cam_filename":rel.as_posix(),"cam_status":status,"leaf_roi_available":"NO",
     "saliency_on_leaf_ratio":"NOT_AVAILABLE","visual_pattern":"","reviewer_comment":"",
     "interpretation_confidence":"UNKNOWN"})
   fig.suptitle(f"{split}: {path}\ntrue={row['true_label']}, predicted={pred}, confidence={float(row['seed42_confidence']):.3f}, wrong seeds={row['wrong_n_seeds']}")
   fig.tight_layout();panel=cam_dir/split/(path.replace("/","__")+"__panel.png")
   panel.parent.mkdir(parents=True,exist_ok=True);fig.savefig(panel,dpi=130);plt.close(fig)
  write_csv(out/"tables/cam_attribution_summary.csv",cam_rows,list(cam_rows[0]))
  write_csv(out/"tables/image_quality_features.csv",quality,list(quality[0]))
  groups=defaultdict(list)
  for row in quality:groups[row["priority_group"]].append(row)
  summary=[]
  for group,rs in groups.items():
   for metric in ("v_mean","s_mean","gray_std","laplacian_variance"):
    vals=[float(r[metric]) for r in rs]
    summary.append({"group":group,"n":len(rs),"metric":metric,"mean":float(np.mean(vals)),
                    "median":float(np.median(vals)),"status":"EXPLORATORY_SMALL_N"})
  write_csv(out/"tables/group_comparison_summary.csv",summary,list(summary[0]))
  holdwrong=[(r["relative_path"],image_data[(r["split"],r["relative_path"])][0]) for r in priority if r["priority_group"]=="healthy_wrong"]
  holdright=[(r["relative_path"],image_data[(r["split"],r["relative_path"])][0]) for r in priority if r["priority_group"]=="healthy_control"]
  disease=[(r["relative_path"],image_data[(r["split"],r["relative_path"])][0]) for r in priority if r["priority_group"]=="disease_to_healthy"]
  montage(holdwrong+holdright,figures/"healthy_originals_blinded.png","Blinded original Healthy review",blinded=True)
  montage(holdright+holdwrong,figures/"healthy_correct_vs_wrong.png","Healthy controls then errors (prediction groups)")
  montage(disease,figures/"disease_to_healthy_examples.png","Disease to Healthy priority cases")
  status=json.loads((out/"execution_status.json").read_text(encoding="utf-8"))
  status.update({"images":"COMPLETE","gradcam":"COMPLETE","priority_images_read":len(image_data),
                 "cam_maps":len(cam_rows),"parity_smoke_images":len(parity),"missing_assets":0})
  (out/"execution_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
  (out/"tables/missing_assets.csv").write_text("asset,reason\n",encoding="utf-8")
  report=out/"REPORT_P0_HEALTHY_ERROR_ANALYSIS.md"
  with report.open("a",encoding="utf-8") as f:
   f.write(f"\n\n## Visual audit addendum\n\nVerified both original image ZIP hashes, read {len(image_data)} priority images, "
           f"passed {len(parity)} classifier/unpooled smoke checks, and generated {len(cam_rows)} CAM maps. "
           "See `figures/gradcam/`, `tables/image_quality_features.csv`, `tables/group_comparison_summary.csv`, "
           "and `tables/parity_smoke.csv`. CAMs are qualitative; no automatic label changes or causal claims.\n")
  checks=[f"{sha_file(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file() and p.name!="CHECKSUMS.sha256"]
  (out/"CHECKSUMS.sha256").write_text("\n".join(checks)+"\n",encoding="utf-8")
  print(json.dumps({"images":len(image_data),"parity_smoke":len(parity),"cam_maps":len(cam_rows),"output":str(out)},indent=2))

if __name__=="__main__":main()
