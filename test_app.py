from streamlit.testing.v1 import AppTest

def test_app_loads_and_controls_recompute():
    app=AppTest.from_file('app.py',default_timeout=60).run()
    assert not app.exception
    app.sidebar.slider[5].set_value(21).run()
    assert not app.exception
    app.sidebar.toggle[0].set_value(False).run()
    assert not app.exception

def test_large_dataset_loads():
    app=AppTest.from_file('app.py',default_timeout=60).run()
    for slider in app.sidebar.slider:
        if slider.label=='History (daily observations)': slider.set_value(180)
        if slider.label=='Countries per business': slider.set_value(40)
        if slider.label=='Lines of business': slider.set_value(7)
    app.run()
    assert not app.exception
    assert app.metric[0].value=='280'
