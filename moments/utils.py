import uuid
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse
from pathlib import Path
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from moments.models import Tag

import jwt
import PIL
from flask import current_app, flash, redirect, request, url_for
from jwt.exceptions import InvalidTokenError
from PIL import Image
from moments.core.extensions import db
from sqlalchemy import func, select


def generate_token(user, operation, expiration=3600, **kwargs):
    payload = {
        'id': user.id,
        'operation': operation.value,
        'exp': datetime.now(timezone.utc) + timedelta(seconds=expiration),
    }
    payload.update(**kwargs)
    return jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')


def parse_token(user, token, operation):
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
    except InvalidTokenError:
        return {}

    if operation.value != payload.get('operation') or user.id != payload.get('id'):
        return {}
    return payload


def rename_image(old_filename):
    ext = Path(old_filename).suffix
    new_filename = uuid.uuid4().hex + ext
    return new_filename


def resize_image(image, filename, base_width):
    ext = Path(filename).suffix
    img = Image.open(image)
    if img.size[0] <= base_width:
        return filename
    w_percent = base_width / float(img.size[0])
    h_size = int(float(img.size[1]) * float(w_percent))
    img = img.resize((base_width, h_size), PIL.Image.LANCZOS)

    filename += current_app.config['MOMENTS_PHOTO_SUFFIXES'][base_width] + ext
    img.save(current_app.config['MOMENTS_UPLOAD_PATH'] / filename, optimize=True, quality=85)
    return filename


def validate_image(filename):
    ext = Path(filename).suffix.lower()
    allowed_extensions = current_app.config['DROPZONE_ALLOWED_FILE_TYPE'].split(',')
    return '.' in filename and ext in allowed_extensions


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


def redirect_back(default='main.index', **kwargs):
    for target in request.args.get('next'), request.referrer:
        if not target:
            continue
        if is_safe_url(target):
            return redirect(target)
    return redirect(url_for(default, **kwargs))


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(f'Error in the {getattr(form, field).label.text} field - {error}')


def get_captions_tags(image_path):
    description = None
    tags = []
    try:
        endpoint = os.environ['VISION_ENDPOINT']
        key = os.environ['VISION_KEY']
    except Exception:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        return (description, tags)

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
    except OSError as e:
        print(f"Error opening image file: {e}")
        return (description, tags)

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS],
        gender_neutral_caption=True,
    )

    
    if result.caption is not None:
        description = result.caption.text
    
    if result.tags is not None:
        for tagObj in result.tags.list:
            tag = db.session.scalar(select(Tag).filter_by(name=tagObj.name))
            if tag is None:
                tag = Tag(name=tagObj.name)
                db.session.add(tag)
                db.session.commit()
            tags.append(tag)
    
    return (description, tags)
