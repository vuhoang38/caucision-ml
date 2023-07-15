from fastapi import FastAPI, File, Form, Request
from fastapi.responses import StreamingResponse

from typing import Annotated

from .config import settings
from .message_queue import initialize_celery
from .mckp import optimize_campaign
from celery import shared_task

from . import repository as repo
from .scylla import Scylla
from .causal_inference import infer_from_project, infer_from_campaign_data
import pickle
import pandas as pd
import numpy as np
import io
import requests
from urllib.parse import urljoin

app = FastAPI()
celery = initialize_celery()


@shared_task(name='train_model', bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 5})
def train_model(self, payload):
    project = repo.find_project(payload['project_id'])
    scylla_db = Scylla()

    df = scylla_db.fetch_table(project.data_id())
    user_effects, est, encoder, identified_estimand, causal_model, categories, training_results = infer_from_project(
        df, project.control_promotion, project.model_type, project.causal_graph
    )

    model_data = {'est': est, 'encoder': encoder, 'identified_estimand': identified_estimand,
                  'causal_model': causal_model, 'categories': categories}
    binary_model_data = pickle.dumps(model_data)

    repo.update_project_model(project.id, binary_model_data)

    scylla_db.save_campaign_estimation(project.campaign_data_id(), user_effects)
    repo.update_project_model_trained(project.id, True)

    unique_values_dict = user_effects.nunique().to_dict()
    type_mappings = {'int64': 'int', 'object': 'text', 'float64': 'float'}

    dtypes = user_effects.dtypes.to_dict()
    data_schema = {k: {'type': type_mappings[str(v)], 'unique_values': unique_values_dict[k]} for k, v in
                   dtypes.items()}

    data = {
        'user_id': str(project.user_id),
        'project_id': str(project.id),
        'data_schema': data_schema,
        'training_results': training_results
    }
    endpoint = urljoin(settings.API_GATEWAY_URL, "/internal/default_campaign")
    requests.post(endpoint, json=data)


@app.get("/")
async def root():
    pass


@app.post("/campaign_data")
async def upload_campaign_data(
        file: Annotated[bytes, File()],
        project_id: Annotated[str, Form()],
        campaign_data_id: Annotated[str, Form()]
):
    project = repo.find_project(project_id)

    csv_io = io.BytesIO(file)
    df = pd.read_csv(csv_io)

    model_data = pickle.loads(project.model)
    identified_estimand = model_data['identified_estimand']
    categories = model_data['categories']
    causal_model = model_data['causal_model']
    est = model_data['est']

    user_effects = infer_from_campaign_data(df, identified_estimand, categories, causal_model, est)
    scylla_db = Scylla()
    scylla_db.save_campaign_estimation(campaign_data_id, user_effects)

    csv_data = io.StringIO()
    user_effects.to_csv(csv_data, index=False)
    csv_data.seek(0)  # Move the file pointer to the beginning of the file

    # Create a StreamingResponse to return the CSV file
    response = StreamingResponse(iter([csv_data.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=data.csv"

    return response


@app.post("/optimize")
async def optimize(request: Request) -> list[str | None]:
    body = await request.json()
    campaign_id = body['campaign_id']
    promotion_costs = body['promotion_costs']
    capacity = body['budget']

    campaign = repo.find_campaign(campaign_id)
    project = repo.find_project(campaign.project_id)

    model_data = pickle.loads(project.model)
    categories: list[str | None] = model_data['categories']

    costs = []
    for category in categories:
        costs.append(promotion_costs[category])

    scylla_db = Scylla()
    df = scylla_db.fetch_table(campaign.data_id())
    category_count = len(categories)

    user_effects = df.iloc[:, 1:(1+(category_count-1)*2)]
    rows = user_effects.values

    zeros_columns = np.zeros((rows.shape[0], 2))
    rows = np.hstack((zeros_columns, rows))

    best_indices = optimize_campaign(rows, capacity, costs)
    categories = np.append(categories, [None])

    return categories[best_indices].tolist()
