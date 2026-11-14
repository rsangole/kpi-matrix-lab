from streamlit.testing.v1 import AppTest

def test_app_loads_and_controls_recompute():
    app=AppTest.from_file('app.py',default_timeout=60).run()
    assert not app.exception
    app.sidebar.slider[5].set_value(21).run()
    assert not app.exception
    app.sidebar.toggle[0].set_value(False).run()
    assert not app.exception
