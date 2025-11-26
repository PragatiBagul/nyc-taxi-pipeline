"""
Start a local Dask Cluster programmatically (useful during development).
"""
import argparse
from dask.distributed import Client,LocalCluster
import time

def main(n_workers:int,
         threads_per_worker:int,
         memory_limit:str,
         dashboard_port:int):
    cluster = LocalCluster(n_workers=n_workers,threads_per_worker=threads_per_worker,memory_limit=memory_limit,dashboard_address=f":{dashboard_port}")
    client = Client(cluster)
    print("Dask scheduler address:",client.scheduler_info().get('address'))
    print("Dashboard:",f"http://{client.scheduler_info().get('address').split('://')[-1].split(':')[0]}:{dashboard_port}")
    print(client)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print('Shutting down cluster...')
        client.close()
        cluster.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=2,dest='threads_per_worker')
    parser.add_argument('--memory-limit',type=str,default='4GB')
    parser.add_argument('--dashboard-port',type=int,default=8787)
    args = parser.parse_args()
    main(args.n_workers,args.threads_per_worker,args.memory_limit,args.dashboard_port)