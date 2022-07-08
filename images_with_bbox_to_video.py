import os

import cv2
from imageio import imread, imwrite
from moviepy.editor import *
import numpy
from rikai.types import Image
from rikai.spark.utils import init_spark_session
from rikai.spark.functions import init


print("Step 1: ML_PREDICT")
spark = init_spark_session(
    dict(
        [
            (
                "spark.rikai.sql.ml.registry.torchhub.impl",
                "ai.eto.rikai.sql.model.torchhub.TorchHubRegistry",
            )
        ]
    )
)
init(spark)

spark.sql("""
CREATE MODEL yolov5m
OPTIONS (device="cpu", batch_size=32)
USING "torchhub:///ultralytics/yolov5:v6.0/yolov5m";
""")
preds = spark.sql("""
from (
  from (
    select image.origin as uri from image.`data/Mojito_10s`
  )
  select uri, to_image(uri) as image
)
select uri, ML_PREDICT(yolov5m, to_image(uri)) as preds
""").collect()


print("Step 2: find the proper bbox to append")
spark.createDataFrame(preds).createOrReplaceTempView("preds_df")
df = spark.sql("""
from (
  from preds_df
  select uri, explode(preds) as pred
)
select uri, pred.box, pred.label, pred.score
where pred.label = 'car'
""")
df.createOrReplaceTempView("df")
result = spark.sql("""
from (
select
  uri, box, label, score,
  row_number() OVER (PARTITION BY uri ORDER BY score desc) AS n 
from df
)
select uri, box, label, score
where n=1
""").collect()


print("Step 3: write to png")
result_dict = {}
for (uri, box, label, score) in result:
    result_dict[uri] = {
	    'box': box,
	    'label': label,
	    'score': score
    }

dir = os.path.realpath("data/Mojito_10s")
tdir = f"/tmp/bbox"
if not os.path.exists(tdir):
    os.mkdir(tdir)

def write_bbox_image(row):
    png = row.value
    for png in os.listdir("data/Mojito_10s"):
        full_png = f"file://{dir}/{png}"
        full_png_bbox = f"{tdir}/{png}"
        if not full_png.endswith("png"):
            continue
        if os.path.exists(full_png_bbox):
            continue
        if full_png in result_dict:
            image = Image(full_png)
            image = image | result_dict[full_png]['box']
            with image.to_image().to_pil() as img:
                img = cv2.cvtColor(numpy.array(img), cv2.COLOR_BGRA2BGR)
                imwrite(f"{tdir}/{png}", img)
        else:
            image = Image(full_png)
            image.save(f"{tdir}/{png}")

df = spark.createDataFrame(os.listdir("data/Mojito_10s"), schema="string")
df.foreach(write_bbox_image)


print("Step 4: generate the video")
clip = ImageSequenceClip(tdir, fps=30)
clip.write_videofile('/tmp/Mojito_10s_bbox.mp4')
