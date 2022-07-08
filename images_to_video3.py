from moviepy.editor import *
from rikai.types import Image
from rikai.spark.utils import init_spark_session
from rikai.spark.functions import init

images_folder = 'data/Mojito_10s'

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

spark.createDataFrame(os.listdir(images_folder), schema="string").createOrReplaceTempView("df")
rows = spark.sql("""
select to_image(concat("data/Mojito_10s/", value)) as image from df order by value asc
""").collect()

images = []
for row in rows:
    images.append(row[0].to_numpy())

clip = ImageSequenceClip(images, fps=30)

clip.write_videofile('data/images_to_video3.mp4')
