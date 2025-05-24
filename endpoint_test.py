import requests

res = requests.post("http://localhost:3000/infer", json={"input": "hello"})
print(res.json())