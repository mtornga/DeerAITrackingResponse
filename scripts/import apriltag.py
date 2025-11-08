import apriltag
from PIL import Image

tag_family = 'tag36h11'
tag_ids = range(6)
detector = apriltag.Detector()
generator = apriltag._get_demo_tag_image  # internal helper

for tag_id in tag_ids:
    img = generator(tag_id, tag_family)
    img.save(f"apriltag_{tag_family}_{tag_id}.png")