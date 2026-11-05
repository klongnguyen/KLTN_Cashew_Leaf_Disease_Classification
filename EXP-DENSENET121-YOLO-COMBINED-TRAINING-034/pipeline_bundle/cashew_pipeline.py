"""Standalone EXP-034 Cashew + YOLO pipeline. Generated from notebook source."""
import json
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps

@keras.utils.register_keras_serializable(package="exp029", name="RandomHSVJitter")
class RandomHSVJitter(layers.Layer):
    """Per-image saturation scaling, hue shift and random grayscale.

    Active only when training=True so the Validation/Test/holdout path stays
    a pure identity, which the smoke test below asserts.
    """

    def __init__(self, saturation_range=(0.6, 1.4), hue_delta=0.03,
                 grayscale_probability=0.10, value_range=(0.0, 255.0),
                 seed=None, **kwargs):
        super().__init__(**kwargs)
        if not (0.0 < saturation_range[0] <= 1.0 <= saturation_range[1]):
            raise ValueError("saturation_range must bracket 1.0")
        if not 0.0 <= hue_delta <= 0.5:
            raise ValueError("hue_delta must be within [0, 0.5]")
        if not 0.0 <= grayscale_probability <= 1.0:
            raise ValueError("grayscale_probability must be within [0, 1]")
        self.saturation_range = tuple(float(v) for v in saturation_range)
        self.hue_delta = float(hue_delta)
        self.grayscale_probability = float(grayscale_probability)
        self.value_range = tuple(float(v) for v in value_range)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        low, high = self.value_range
        span = high - low
        normalized = tf.clip_by_value((inputs - low) / span, 0.0, 1.0)
        batch = tf.shape(normalized)[0]
        saturation = keras.random.uniform(
            (batch, 1, 1), self.saturation_range[0], self.saturation_range[1],
            seed=self.seed_generator)
        grayscale_draw = keras.random.uniform(
            (batch, 1, 1), 0.0, 1.0, seed=self.seed_generator)
        saturation = tf.where(grayscale_draw < self.grayscale_probability,
                              tf.zeros_like(saturation), saturation)
        hue_shift = keras.random.uniform(
            (batch, 1, 1), -self.hue_delta, self.hue_delta, seed=self.seed_generator)
        hsv = tf.image.rgb_to_hsv(normalized)
        hue = tf.math.floormod(hsv[..., 0] + hue_shift, 1.0)
        sat = tf.clip_by_value(hsv[..., 1] * saturation, 0.0, 1.0)
        value = hsv[..., 2]
        converted = tf.image.hsv_to_rgb(tf.stack([hue, sat, value], axis=-1))
        return tf.clip_by_value(converted * span + low, low, high)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"saturation_range": list(self.saturation_range),
                       "hue_delta": self.hue_delta,
                       "grayscale_probability": self.grayscale_probability,
                       "value_range": list(self.value_range),
                       "seed": self.seed})
        return config



class CashewPipeline:
    def __init__(self,bundle_dir):
        self.bundle_dir=Path(bundle_dir)
        self.config=json.loads((self.bundle_dir/"frozen_config.json").read_text(encoding="utf-8"))
        self.classes=self.config["classes"]
        self.detector_classes=self.config["detector_classes"]
        self.models=[keras.models.load_model(self.bundle_dir/f"seed_{seed}"/"deployment_model.keras")
                     for seed in self.config["seeds"]]
        self.stackers=joblib.load(self.bundle_dir/"stacker.joblib")
        self.rule=json.loads((self.bundle_dir/"decision.json").read_text(encoding="utf-8"))["selected_default_rule"]
        if self.config.get("dry_run_stub_detector"):
            self.detector=None
        else:
            from ultralytics import YOLO
            self.detector=YOLO(str(self.bundle_dir/"best.pt"))

    @staticmethod
    def _iou(a,b):
        x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
        return inter/(aa+bb-inter) if aa+bb>inter else 0.0

    def _boxes(self,image,path):
        height,width=image.shape[:2]
        if self.detector is None:
            if path is None or np.std(image)==0:
                return [],(height,width)
            name=Path(path).parent.name
            cls=(self.detector_classes.index(name) if name in self.detector_classes else 0)
            return ([{"cls":self.detector_classes[cls],"conf":.75,
                     "xyxy":[width*.3,height*.3,width*.7,height*.7]}],(height,width))
        source=str(path) if path is not None else image
        result=self.detector.predict(source,conf=.05,iou=.7,imgsz=640,rect=False,verbose=False)[0]
        boxes=[{"cls":self.detector_classes[int(c)],"conf":float(q),"xyxy":xy.tolist()}
               for c,q,xy in zip(result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy(),
                                 result.boxes.xyxy.cpu().numpy())]
        return boxes,tuple(result.orig_shape)

    def predict(self,image_path_or_array):
        path=Path(image_path_or_array) if isinstance(image_path_or_array,(str,Path)) else None
        if path is not None:
            # Match make_dataset: tf.io.decode_image does not apply PIL EXIF transpose.
            image=tf.io.decode_image(tf.io.read_file(str(path)),channels=3,
                                     expand_animations=False).numpy()
        else:
            image=np.asarray(image_path_or_array)
            if image.ndim==2: image=np.repeat(image[...,None],3,axis=2)
            elif image.ndim==3 and image.shape[2]==1: image=np.repeat(image,3,axis=2)
            elif image.ndim==3 and image.shape[2]==4: image=image[:,:,:3]
            if image.ndim!=3 or image.shape[2]!=3: raise ValueError("Expected RGB or grayscale image")
            image=np.clip(image,0,255).astype(np.uint8)
        resized=tf.image.resize(tf.cast(image,tf.float32),(256,256)).numpy()[None,...]
        p=np.mean([model.predict(resized,verbose=0)[0] for model in self.models],axis=0)
        boxes,(h,w)=self._boxes(image,path)
        evidence={"max_conf":{},"n_boxes":{},"area_frac":{}}
        for cls in self.detector_classes:
            subset=[b for b in boxes if b["cls"]==cls]
            evidence["max_conf"][cls]=max([b["conf"] for b in subset],default=0.0)
            evidence["n_boxes"][cls]=len(subset)
            evidence["area_frac"][cls]=min(sum((b["xyxy"][2]-b["xyxy"][0])*
                 (b["xyxy"][3]-b["xyxy"][1]) for b in subset)/(w*h),1.0)
        feature=np.array(list(p)+[evidence[prefix][cls] for prefix in
                     ("max_conf","n_boxes","area_frac") for cls in self.detector_classes],dtype=float)
        assert self.rule == "F3"
        model=self.stackers["F3"]
        fitted=model.predict_proba(feature[None,:])[0]
        p=np.zeros(len(self.classes));p[model[-1].classes_]=fitted
        temperature=self.config["temperature"][self.rule]
        p=np.clip(p,1e-9,1)**(1/temperature);p=p/p.sum()
        order=np.argsort(-p);label=self.classes[order[0]]
        abstain=bool(p[order[0]]<self.config["abstain_thresholds"][self.rule])
        shown=[]
        if not abstain and label in self.detector_classes:
            for box in sorted((b for b in boxes if b["conf"]>=self.config["box_conf"]),
                              key=lambda b:-b["conf"]):
                if all(box["cls"]!=old["cls"] or self._iou(box["xyxy"],old["xyxy"])<=.5 for old in shown):
                    shown.append(box)
                if len(shown)>=60: break
        return {"label":label,"confidence":float(p[order[0]]),
                "top2":[(self.classes[i],float(p[i])) for i in order[:2]],
                "abstain":abstain,"probs":{name:float(p[i]) for i,name in enumerate(self.classes)},
                "evidence":evidence,"boxes":shown,"rule":self.rule,"version":"EXP-034",
                "action":"cần chuyên gia kiểm tra" if abstain else ("không phát hiện bệnh" if label in ("healthy","not_cashew_leaf") else "predict")}
