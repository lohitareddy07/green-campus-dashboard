from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=120).run()
print("exceptions:", at.exception)
print("metrics:", [m.value for m in at.metric])
