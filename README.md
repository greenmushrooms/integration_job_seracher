for local runs 
export $(grep -v '^#' .env | xargs)
prefect deploy --name job-search-deployment
prefect deploy --name job-search-deployment

git submodule update --remote data__job_searcher
