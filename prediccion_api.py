from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "regresion_balanceada.joblib"
)

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


SERVICIOS = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]


def enriquecer_cliente(cliente):
    cliente = cliente.copy()
    tenure = cliente["tenure"]

    if tenure <= 6:
        grupo = "0-6m"
    elif tenure <= 12:
        grupo = "7-12m"
    elif tenure <= 24:
        grupo = "13-24m"
    elif tenure <= 48:
        grupo = "25-48m"
    else:
        grupo = "49-72m"

    cliente["tenure_grupo"] = grupo
    cliente["num_servicios"] = sum(
        cliente[s] == "Yes" for s in SERVICIOS
    )
    cliente["tiene_internet"] = (
        "No" if cliente["InternetService"] == "No" else "Yes"
    )
    cliente["pago_automatico"] = (
        "Yes" if "automatic" in cliente["PaymentMethod"] else "No"
    )
    cliente["cargo_promedio_hist"] = (
        cliente["TotalCharges"] / tenure
        if tenure > 0 else cliente["MonthlyCharges"]
    )
    cliente["delta_cargo"] = (
        cliente["MonthlyCharges"] - cliente["cargo_promedio_hist"]
    )

    return cliente


def mostrar_prediccion():
    st.subheader("Predicción individual de abandono")
    st.write(
        "Ingrese la información del cliente para consultar "
        "la probabilidad de abandono mediante el modelo desplegado."
    )

    with st.form("formulario_prediccion"):
        st.markdown("#### Perfil del cliente")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Género", ["Female", "Male"])
            senior = st.selectbox("Adulto mayor", ["No", "Yes"])

        with c2:
            partner = st.selectbox("Tiene pareja", ["No", "Yes"])
            dependents = st.selectbox("Tiene dependientes", ["No", "Yes"])

        with c3:
            tenure = st.number_input(
                "Antigüedad (meses)", 0, 72, 12
            )

        st.markdown("#### Servicios contratados")

        a, b, c = st.columns(3)

        with a:
            phone = st.selectbox("Servicio telefónico", ["Yes", "No"])
            multiple = st.selectbox("Múltiples líneas", ["No", "Yes"])
            internet = st.selectbox(
                "Servicio de internet", ["DSL", "Fiber optic", "No"]
            )

        with b:
            security = st.selectbox("Seguridad en línea", ["No", "Yes"])
            backup = st.selectbox("Respaldo en línea", ["No", "Yes"])
            protection = st.selectbox(
                "Protección de dispositivos", ["No", "Yes"]
            )

        with c:
            support = st.selectbox("Soporte técnico", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming películas", ["No", "Yes"])

        st.markdown("#### Contrato y facturación")

        d, e = st.columns(2)

        with d:
            contract = st.selectbox(
                "Tipo de contrato",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Facturación electrónica", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Método de pago",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check"
                ]
            )

        with e:
            monthly = st.number_input(
                "Cargo mensual (USD)", min_value=0.0,
                value=70.0, step=1.0
            )
            total = st.number_input(
                "Cargo total acumulado (USD)", min_value=0.0,
                value=840.0, step=10.0
            )

        enviar = st.form_submit_button(
            "Calcular probabilidad de abandono",
            type="primary"
        )

    if enviar:
        if phone == "No" and multiple == "Yes":
            st.error(
                "Un cliente sin servicio telefónico "
                "no puede tener múltiples líneas."
            )
            return

        if internet == "No" and any(
            x == "Yes" for x in [
                security, backup, protection, support, tv, movies
            ]
        ):
            st.error(
                "Un cliente sin internet no puede tener "
                "servicios adicionales de internet."
            )
            return

        cliente = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total)
        }

        cliente = enriquecer_cliente(cliente)

        try:
            modelo = cargar_modelo()
            datos = pd.DataFrame([cliente])
            datos = datos[list(modelo.feature_names_in_)]

            prediccion = int(modelo.predict(datos)[0])
            probabilidad = float(
                modelo.predict_proba(datos)[0, 1]
            )

            st.success("Predicción realizada correctamente")

            r1, r2 = st.columns(2)
            r1.metric(
                "Probabilidad estimada de abandono",
                f"{probabilidad:.1%}"
            )
            r2.metric(
                "Clasificación del modelo",
                "Abandono" if prediccion == 1 else "No abandono"
            )

            st.progress(
                min(max(probabilidad, 0.0), 1.0)
            )

            st.caption(
                "Resultado orientativo del modelo. "
                "No garantiza que el cliente abandone el servicio."
            )

        except Exception as error:
            st.error(f"No fue posible generar la predicción: {error}")
