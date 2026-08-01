import urllib.request
import json
import zipfile
import io

try:
    # 1. Get the latest runs
    url = "https://api.github.com/repos/tamimystic/Chokkhu-PyPi-Package/actions/runs"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req)
    data = json.loads(res.read())
    
    # 2. Get the specific run ID for "release v0.3.1"
    run_id = None
    for run in data['workflow_runs']:
        if "0.3.1" in run['head_commit']['message']:
            run_id = run['id']
            break
            
    if not run_id:
        print("Run ID for 0.3.1 not found.")
    else:
        print(f"Run ID: {run_id}")
        
        # 3. Get the jobs for this run
        jobs_url = f"https://api.github.com/repos/tamimystic/Chokkhu-PyPi-Package/actions/runs/{run_id}/jobs"
        req = urllib.request.Request(jobs_url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        jobs_data = json.loads(res.read())
        
        for job in jobs_data['jobs']:
            print(f"{job['name']}: {job['conclusion']}")
            if job['conclusion'] == 'failure' and '3.10' in job['name']:
                print(f"Fetching logs for failed job: {job['name']} ({job['id']})")
                log_url = f"https://api.github.com/repos/tamimystic/Chokkhu-PyPi-Package/actions/jobs/{job['id']}/logs"
                try:
                    log_req = urllib.request.Request(log_url, headers={'User-Agent': 'Mozilla/5.0'})
                    log_res = urllib.request.urlopen(log_req)
                    logs = log_res.read().decode('utf-8')
                    print("--- ERROR LOG TAIL ---")
                    print("\n".join(logs.splitlines()[-50:]))
                    print("----------------------")
                except Exception as e:
                    print(f"Failed to fetch log for job {job['id']}: {e}")

except Exception as e:
    print(f"Error: {e}")
