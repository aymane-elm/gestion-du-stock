import os
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy import exc as sa_exc
from sqlalchemy.engine import Engine

# =========================
# CONFIG
LOGO_PATH = "Qwintal_logo.png"

# =========================
# === [REF:LOGO-FAVICON] Favicon de la page
from PIL import Image
try:
    st.set_page_config(
        page_title="Gestion de stock & fabrication",
        layout="wide",
        page_icon=Image.open(LOGO_PATH)  # favicon
    )
except Exception:
    st.set_page_config(page_title="Gestion de stock & fabrication", layout="wide", page_icon="📘")

st.title("Gestion de stock & fabrication")

DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_gW2a0Hlfzpxn@ep-divine-scene-agixk2f3-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# === [REF:LOGO-FILES] Chemin du logo


# =========================
# SQL – Connexion & seed responsables
# =========================
@st.cache_resource
def get_engine() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True)

engine = get_engine()


def run_seed_responsables():
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO responsables(name) VALUES
              ('Aymane'), ('Joslain'), ('Lise'), ('Robin')
            ON CONFLICT (name) DO NOTHING
        """))

run_seed_responsables()

# =========================
# HELPERS SQL
# =========================

def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        res = conn.execute(text(sql), params or {})
        rows = res.mappings().all()
        if not rows:
            return pd.DataFrame(columns=list(res.keys()))  # ← garde les noms de colonnes
        return pd.DataFrame(rows)



def execute(sql: str, params: Dict[str, Any] | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


# =========================
# LOGIQUE MÉTIER – RÉFÉRENTIELS
# =========================

def get_responsables() -> List[str]:
    df = fetch_df("SELECT name FROM responsables ORDER BY name")
    return df["name"].astype(str).tolist()


def get_clients() -> pd.DataFrame:
    # id (uuid), client_name, type, phone, email, notes
    return fetch_df("SELECT * FROM clients ORDER BY client_name")


def insert_client_return_id(name: str, ctype: str = "Passage", phone: str | None = None,
                            email: str | None = None, notes: str | None = None) -> str:
    with engine.begin() as conn:
        row = conn.execute(text(
            """
            INSERT INTO clients (client_name, type, phone, email, notes)
            VALUES (:name, :type, :phone, :email, :notes)
            RETURNING id
            """
        ), {"name": name.strip(), "type": ctype, "phone": phone, "email": email, "notes": notes}).mappings().first()
        return str(row["id"])  # uuid


# remplace intégralement la fonction actuelle
def delete_clients(ids: List[str]) -> int:
    if not ids:
        return 0

    # Colonne id est de type UUID → on cast
    ids_uuid = [uuid.UUID(x) for x in ids]

    # expanding IN :ids → SQLAlchemy génère automatiquement (.., .., ..)
    stmt = text("DELETE FROM clients WHERE id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )

    with engine.begin() as conn:
        res = conn.execute(stmt, {"ids": ids_uuid})
        return res.rowcount



# =========================
# LOGIQUE MÉTIER – STOCK & MOUVEMENTS
# =========================

def get_stock() -> pd.DataFrame:
    df = fetch_df(
        """
        SELECT sku, name, unit, category,
               COALESCE(reorder_point,0) AS reorder_point,
               COALESCE(qty_on_hand,0)   AS qty_on_hand,
               description
        FROM stock
        ORDER BY sku
        """
    )
    df["reorder_point"] = pd.to_numeric(df["reorder_point"], errors="coerce").fillna(0.0)
    df["qty_on_hand"]   = pd.to_numeric(df["qty_on_hand"], errors="coerce").fillna(0.0)
    return df


def upsert_stock_row(row: Dict[str, Any]) -> None:
    execute(
        """
        INSERT INTO stock (sku, name, unit, category, reorder_point, qty_on_hand, description)
        VALUES (:sku, :name, :unit, :category, :reorder_point, :qty_on_hand, :description)
        ON CONFLICT (sku) DO UPDATE SET
          name = EXCLUDED.name,
          unit = EXCLUDED.unit,
          category = EXCLUDED.category,
          reorder_point = EXCLUDED.reorder_point,
          qty_on_hand = EXCLUDED.qty_on_hand,
          description = EXCLUDED.description
        """,
        row,
    )


def add_stock_item(sku: str, name: str, unit: str, category: str,
                   reorder_point: float, qty_on_hand: float, description: str | None):
    upsert_stock_row({
        "sku": sku, "name": name, "unit": unit, "category": category,
        "reorder_point": float(reorder_point or 0),
        "qty_on_hand": float(qty_on_hand or 0),
        "description": description
    })


def record_movement_and_update(sku: str, move_type: str, qty: float,
                               ref: str, location: str, responsable: str,
                               mo_uuid: str | None = None) -> float:
    if qty < 0:
        raise ValueError("La quantité doit être >= 0")
    delta = qty if move_type == "IN" else -qty
    with engine.begin() as conn:
        r = conn.execute(text("SELECT qty_on_hand FROM stock WHERE sku=:sku FOR UPDATE"), {"sku": sku}).first()
        if r is None:
            raise ValueError(f"SKU introuvable: {sku}")
        current = float(r[0] or 0)
        new_val = current + delta
        if new_val < 0:
            raise ValueError("Stock négatif interdit")
        conn.execute(text("UPDATE stock SET qty_on_hand=:q WHERE sku=:sku"), {"q": new_val, "sku": sku})
        conn.execute(text(
            """
            INSERT INTO mouvements(date, sku, type, qty, ref, location, mo_id, responsable)
            VALUES (now(), :sku, :type, :qty, :ref, :loc, :mo, :resp)
            """
        ), {"sku": sku, "type": move_type, "qty": float(qty), "ref": ref,
            "loc": location, "mo": mo_uuid, "resp": responsable})
        return new_val


# =========================
# LOGIQUE – FILTRES SQL ROBUSTES (IN dynamique)
# =========================

def _expand_in_clause(sql: str, field: str, values: List[str], param_prefix: str, params: Dict[str, Any]) -> str:
    placeholders = []
    for i, val in enumerate(values):
        k = f"{param_prefix}_{i}"
        placeholders.append(f":{k}")
        params[k] = val
    sql += f" AND {field} IN ({', '.join(placeholders)})"
    return sql


def get_mouvements_filtered(d_from: date, d_to: date, types: List[str],
                            sku_like: str | None, responsable: str | None) -> pd.DataFrame:
    q = (
        """
        SELECT id, date, sku, type, qty, ref, location, mo_id, responsable
        FROM mouvements
        WHERE date::date BETWEEN :dfrom AND :dto
        """
    )
    params: Dict[str, Any] = {"dfrom": d_from, "dto": d_to}
    if types:
        # enum movement_type => valeurs 'IN'/'OUT'
        norm = [str(t).upper() for t in types]
        q = _expand_in_clause(q, "type", norm, "type", params)
    if sku_like and sku_like.strip():
        q += " AND sku ILIKE :sk"
        params["sk"] = f"%{sku_like.strip()}%"
    if responsable and responsable != "(Tous)":
        q += " AND responsable = :resp"
        params["resp"] = responsable
    q += " ORDER BY date DESC"
    return fetch_df(q, params)


def get_fabrications_filtered(d_from: date, d_to: date, products: List[str],
                              status: str | None, client_like: str | None) -> pd.DataFrame:
    q = (
        """
        SELECT f.mo_id, f.date, f.due_date, f.product, f.qty, f.status, f.ref, f.responsable,
               c.client_name
        FROM fabrications f
        LEFT JOIN clients c ON c.id = f.client_id
        WHERE f.date::date BETWEEN :dfrom AND :dto
        """
    )
    params: Dict[str, Any] = {"dfrom": d_from, "dto": d_to}
    if products:
        q = _expand_in_clause(q, "f.product", products, "prod", params)
    if status and status != "(Tous)":
        q += " AND f.status = :st"
        params["st"] = status
    if client_like and client_like.strip():
        q += " AND c.client_name ILIKE :cl"
        params["cl"] = f"%{client_like.strip()}%"
    q += " ORDER BY COALESCE(f.due_date, CURRENT_DATE) ASC, f.date DESC"
    return fetch_df(q, params)


# =========================
# LOGIQUE – BOM & OF
# =========================

def get_bom(product: str) -> pd.DataFrame:
    if product == "GMQ ONE":
        return fetch_df(
            "SELECT component_sku AS componentsku, qty_per_unit AS qtyperunit, description FROM bom_gmq_one"
        )
    elif product == "GMQ LIVE":
        return fetch_df(
            "SELECT component_sku AS componentsku, qty_per_unit AS qtyperunit, description FROM bom_gmq_live"
        )
    else:
        raise ValueError("Produit inconnu")


def check_availability_sql(product: str, qty_make: float) -> Tuple[pd.DataFrame, bool]:
    bom = get_bom(product)
    stock = get_stock()[["sku","qty_on_hand"]].rename(columns={"sku": "componentsku"})
    df = bom.merge(stock, on="componentsku", how="left")
    df["qtyperunit"] = pd.to_numeric(df["qtyperunit"], errors="coerce").fillna(0.0)
    df["qty_on_hand"] = pd.to_numeric(df["qty_on_hand"], errors="coerce").fillna(0.0)
    df["Besoin (total)"] = df["qtyperunit"] * float(qty_make or 0)
    df["Stock dispo"] = df["qty_on_hand"]
    df["Manque"] = (df["Besoin (total)"] - df["Stock dispo"]).clip(lower=0.0)
    out = df.rename(columns={
        "componentsku": "ComponentSKU",
        "qtyperunit": "QtyPerUnit",
        "description": "Description",
    })[["ComponentSKU", "QtyPerUnit", "Besoin (total)", "Stock dispo", "Manque", "Description"]]
    out = out.sort_values(["Manque", "ComponentSKU"], ascending=[False, True]).reset_index(drop=True)
    ok = float(out["Manque"].sum()) == 0.0
    return out, ok


def post_fabrication(product: str, qty_make: float, due_date: date,
                     ref: str, responsable: str, client_id: str | None) -> str:
    req_df, ok = check_availability_sql(product, qty_make)
    if not ok:
        raise ValueError("Stock insuffisant pour poster l'OF")

    mo_uuid = str(uuid.uuid4())  # UUID valide pour colonne UUID
    fin_sku = "GMQ-ONE" if product == "GMQ ONE" else "GMQ-LIVE"

    with engine.begin() as conn:
        # 1) Insert fabrication
        conn.execute(text(
            """
            INSERT INTO fabrications (mo_id, date, due_date, product, qty, status, ref, responsable, client_id)
            VALUES (:mo, now(), :due, :prod, :qty, 'Posté', :ref, :resp, :client_id)
            """
        ), {"mo": mo_uuid, "due": due_date, "prod": product, "qty": float(qty_make),
            "ref": ref, "resp": responsable, "client_id": client_id})

        # 2) Composants OUT
        for _, r in req_df.iterrows():
            comp_sku = r["ComponentSKU"]
            need = float(r["Besoin (total)"])
            # MAJ stock composant
            conn.execute(text("UPDATE stock SET qty_on_hand = COALESCE(qty_on_hand,0) - :q WHERE sku=:s"),
                         {"q": need, "s": comp_sku})
            # Mouvement
            conn.execute(text(
                """
                INSERT INTO mouvements(date, sku, type, qty, ref, location, mo_id, responsable)
                VALUES (now(), :sku, 'OUT', :qty, :ref, 'PROD', :mo, :resp)
                """
            ), {"sku": comp_sku, "qty": need, "ref": ref, "mo": mo_uuid, "resp": responsable})

        # 3) Produit fini IN
        conn.execute(text("UPDATE stock SET qty_on_hand = COALESCE(qty_on_hand,0) + :q WHERE sku=:s"),
                     {"q": float(qty_make), "s": fin_sku})
        conn.execute(text(
            """
            INSERT INTO mouvements(date, sku, type, qty, ref, location, mo_id, responsable)
            VALUES (now(), :sku, 'IN', :qty, :ref, 'STOCK', :mo, :resp)
            """
        ), {"sku": fin_sku, "qty": float(qty_make), "ref": ref, "mo": mo_uuid, "resp": responsable})

    return mo_uuid


# =========================
# UTILS EXPORT
# =========================

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# =========================
# UI – TABS
# =========================

# [REF:TABS-DEFINE]
tab_dash, tab_moves, tab_mo, tab_stock, tab_invent, tab_clients, tab_bom = st.tabs([
    "Aperçu", "Mouvements", "Ordres de fabrication", "Stock",
    "Inventaire", "Clients", "BOM GMQ"
])


# ---- DASHBOARD
with tab_dash:
    st.subheader("Aperçu")
    stock_df = get_stock()
    total_skus = int(len(stock_df))
    total_qty = float(stock_df["qty_on_hand"].sum())
    mo_posted = int(fetch_df("SELECT COUNT(*) AS c FROM fabrications WHERE status='Posté'")[["c"]].iat[0, 0])
    low = stock_df[stock_df["qty_on_hand"] < stock_df["reorder_point"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles (SKU)", f"{total_skus}")
    c2.metric("Qté totale", f"{total_qty:,.0f}")
    c3.metric("OF postés", f"{mo_posted}")
    c4.metric("Sous seuil", f"{len(low)}")
    st.markdown("#### Sous le seuil")
    st.dataframe(low if not low.empty else stock_df.iloc[0:0], use_container_width=True)

# ---- MOUVEMENTS (form + filtres + tableau)
with tab_moves:
    st.header("Mouvements")
    resp_list = get_responsables() or ["Aymane", "Joslain", "Lise", "Robin"]
    stock_df = get_stock()

    st.subheader("Ajouter un mouvement")
    with st.form("mv_form"):
        col_a, col_b = st.columns(2)
        sku = col_a.selectbox("SKU", stock_df["sku"].astype(str).tolist())
        responsable = col_b.selectbox("Responsable", resp_list, index=0)
        move_type = st.radio("Type", ["IN", "OUT"], horizontal=True)
        qty = st.number_input("Quantité", min_value=0.0, step=1.0)
        ref = st.text_input("Référence", value="MANUAL")
        loc = st.text_input("Emplacement", value="ENTREPOT")
        submitted = st.form_submit_button("Enregistrer")
        if submitted:
            if qty <= 0:
                st.error("La quantité doit être > 0.")
            else:
                try:
                    new_qty = record_movement_and_update(sku, move_type, qty, ref, loc, responsable)
                    st.success(f"Mouvement {move_type} enregistré. Nouveau stock {sku} = {new_qty}")
                    st.toast("Mouvement enregistré")
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.subheader("Historique des mouvements")
    mv_all = fetch_df("SELECT MIN(date)::date AS dmin, MAX(date)::date AS dmax FROM mouvements")
    default_from = (mv_all.get("dmin").iat[0] if not mv_all.empty else None) or (date.today() - timedelta(days=30))
    default_to   = (mv_all.get("dmax").iat[0] if not mv_all.empty else None) or date.today()

    # === [REF:MV-FILTERS] Historique des mouvements ===
    c1, c2, c3 = st.columns(3)
    d_from = c1.date_input("Du", value=default_from, key="mv_hist_from")           # REF:MV_FROM
    d_to   = c2.date_input("Au", value=default_to,   key="mv_hist_to")             # REF:MV_TO
    types  = c3.multiselect("Type", options=["IN","OUT"], 
                            default=["IN","OUT"], key="mv_hist_types")             # REF:MV_TYPES
    
    c4, c5 = st.columns(2)
    sku_filter = c4.text_input("Filtre SKU (contient)", "", key="mv_hist_sku")     # REF:MV_SKU
    resp_opts  = ["(Tous)"] + resp_list
    resp_pick  = c5.selectbox("Responsable", resp_opts, index=0, key="mv_hist_resp")  # REF:MV_RESP


    mv_view = get_mouvements_filtered(d_from, d_to, types, sku_filter, resp_pick if resp_pick != "(Tous)" else None)
    st.dataframe(mv_view, use_container_width=True)
    

# === ACCESSOIRES (helpers) ===


def get_accessory_catalog() -> pd.DataFrame:
    """
    Catalogue d’accessoires depuis 'stock'.
    Essaie d'abord category ∈ Accessoire/Accessories, sinon retombe sur tout le stock.
    Renvoie colonnes normalisées: id (=sku), item_name (=name), unit.
    """
    df = fetch_df("""
        SELECT sku::text AS id, name::text AS item_name, COALESCE(unit::text,'') AS unit
        FROM stock
        WHERE lower(category) IN ('accessoire','accessoires','accessory','accessories')
        ORDER BY name
    """)
    if not df.empty:
        return df
    return fetch_df("""
        SELECT sku::text AS id, name::text AS item_name, COALESCE(unit::text,'') AS unit
        FROM stock
        ORDER BY name
    """)

def save_of_accessories(of_id: str, rows: list[dict]) -> int:
    """
    Remplace les accessoires d’un OF (DELETE + INSERT).
    rows = [{component_sku:str, qty:float, notes:str}]
    """
    rows = rows or []
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM fabrication_accessories WHERE of_id = :of_id"), {"of_id": of_id})
        if rows:
            stmt = text("""
                INSERT INTO fabrication_accessories (of_id, component_sku, qty, notes)
                VALUES (:of_id, :component_sku, :qty, :notes)
            """)
            payload = [{"of_id": of_id, **r} for r in rows]
            conn.execute(stmt, payload)
    return len(rows)



# ---- ORDRES DE FABRICATION
with tab_mo:
    st.header("Ordres de fabrication")
    resp_list = get_responsables() or ["Aymane", "Joslain", "Lise", "Robin"]
    clients_df = get_clients()

    st.subheader("Créer un OF")
    with st.form("mo_form"):
        col1, col2 = st.columns(2)
        product = col1.selectbox("Produit fini", ["GMQ ONE", "GMQ LIVE"])
        responsable = col2.selectbox("Responsable", resp_list, index=0)
        col3, col4, col5 = st.columns([1, 1, 2])
        qty_make = col3.number_input("Quantité à produire", min_value=0.0, step=1.0)
        due_date = col4.date_input("Date d'échéance", value=date.today() + timedelta(days=7))
        ref = col5.text_input("Référence OF", value="OF-AUTO")

        st.markdown("**Client associé à l'OF**")
        id_to_label = {"NONE": "(aucun)"}
        if not clients_df.empty:
            for r in clients_df.itertuples(index=False):
                id_to_label[str(r.id)] = str(r.client_name)
        id_to_label["NEW"] = "Client de passage (saisie)"
        options = list(id_to_label.keys())
        selected = st.selectbox("Client", options=options, index=0, format_func=lambda k: id_to_label[k])
        client_free = None
        if selected == "NEW":
            client_free = st.text_input("Nom du client (passage)", value="")

        # === NEW: Accessoires pour l'OF ===
        st.markdown("**Accessoires (optionnel)**")
        acc_catalog = get_accessory_catalog()
        acc_id_to_name = dict(zip(acc_catalog["id"].astype(str), acc_catalog["item_name"]))
        acc_id_to_unit = dict(zip(acc_catalog["id"].astype(str), acc_catalog["unit"]))

        # état local du formulaire
        if "of_acc_df_form" not in st.session_state:
            st.session_state["of_acc_df_form"] = pd.DataFrame(
                columns=["component_sku", "item_name", "unit", "qty", "notes"]
            )

        with st.expander("➕ Ajouter des accessoires depuis le stock"):
            picked = st.multiselect(
                "Accessoires (SKU)",
                options=acc_catalog["id"].astype(str).tolist(),
                format_func=lambda sku: f"{acc_id_to_name.get(sku,'??')} — {sku}",
                key="of_acc_picked",
            )
            default_qty = st.number_input("Quantité par défaut (accessoire)", min_value=0.0, value=1.0, step=1.0, key="of_acc_default_qty")
            if st.form_submit_button("Ajouter à la liste d’accessoires"):
                cur = st.session_state["of_acc_df_form"].copy()
                existing = set(cur["component_sku"].astype(str)) if not cur.empty else set()
                to_add = [sku for sku in picked if sku not in existing]
                if to_add:
                    add_rows = pd.DataFrame({
                        "component_sku": to_add,
                        "item_name": [acc_id_to_name.get(sku, "??") for sku in to_add],
                        "unit": [acc_id_to_unit.get(sku, "") for sku in to_add],
                        "qty": [default_qty for _ in to_add],
                        "notes": ["" for _ in to_add],
                    })
                    st.session_state["of_acc_df_form"] = pd.concat([cur, add_rows], ignore_index=True)
                    st.success(f"{len(to_add)} accessoire(s) ajouté(s).")
                else:
                    st.info("Aucun nouvel accessoire.")

        st.data_editor(
            st.session_state["of_acc_df_form"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "component_sku": st.column_config.TextColumn("SKU accessoire"),
                "item_name": st.column_config.TextColumn("Nom", disabled=True),
                "unit": st.column_config.TextColumn("Unité", disabled=True),
                "qty": st.column_config.NumberColumn("Quantité", min_value=0.0, step=0.1),
                "notes": st.column_config.TextColumn("Notes"),
            },
            key="of_acc_editor_form",
        )
        # === /NEW ===

        cver, cpost = st.columns(2)
        verify_clicked = cver.form_submit_button("Vérifier l'OF")
        post_clicked = cpost.form_submit_button("Poster l'OF")

        if verify_clicked or post_clicked:
            if qty_make <= 0:
                st.error("La quantité doit être > 0.")
            else:
                req_df, ok = check_availability_sql(product, qty_make)
                st.markdown("#### Besoins vs stock (BOM)")
                st.dataframe(req_df, use_container_width=True)

                if not ok:
                    manques = req_df.loc[req_df["Manque"] > 0, ["ComponentSKU", "Manque"]]
                    manques_str = ", ".join([
                        f"{r.ComponentSKU} (-{r.Manque:.0f})" if float(r.Manque).is_integer() else f"{r.ComponentSKU} (-{r.Manque})"
                        for r in manques.itertuples()
                    ])
                    st.error(f"Stock insuffisant. Composants manquants : {manques_str}")
                else:
                    st.success("Stock OK pour l'OF.")

                if post_clicked and ok:
                    # client_id final
                    client_id = None
                    if selected == "NEW":
                        name = (client_free or "").strip()
                        client_id = insert_client_return_id(name, ctype="Passage") if name else None
                    elif selected not in ("NONE", "NEW"):
                        client_id = selected

                    try:
                        mo_id = post_fabrication(product, qty_make, due_date, ref, responsable, client_id)

                        # === NEW: enregistrer les accessoires liés à l'OF ===
                        acc_df = st.session_state.get("of_acc_df_form", pd.DataFrame())
                        if not acc_df.empty:
                            # filtre: SKU existant dans le catalogue et qty>0
                            valid = set(acc_catalog["id"].astype(str))
                            work = acc_df.copy()
                            work = work[work["component_sku"].astype(str).isin(valid)]
                            work = work[pd.to_numeric(work["qty"], errors="coerce").fillna(0) > 0]
                            rows = [{
                                "component_sku": str(r["component_sku"]),
                                "qty": float(r["qty"]),
                                "notes": str(r.get("notes","") or ""),
                            } for _, r in work.iterrows()]
                            if rows:
                                save_of_accessories(mo_id, rows)
                        # reset accessoires du formulaire après post
                        st.session_state["of_acc_df_form"] = pd.DataFrame(columns=["component_sku","item_name","unit","qty","notes"])
                        # === /NEW ===

                        st.success(f"OF {mo_id} posté par {responsable} (échéance {due_date:%Y-%m-%d}).")
                        st.toast("OF posté")
                    except Exception as e:
                        st.error(str(e))


    st.divider()
    st.subheader("Liste des ordres de fabrication")
    fab_mm = fetch_df("SELECT MIN(date)::date AS dmin, MAX(date)::date AS dmax FROM fabrications")
    default_f_from = (fab_mm.get("dmin").iat[0] if not fab_mm.empty else None) or (date.today() - timedelta(days=30))
    default_f_to   = (fab_mm.get("dmax").iat[0] if not fab_mm.empty else None) or date.today()

        # === [REF:FAB-LIST-FILTERS] Liste des OF ===
    f1, f2, f3 = st.columns(3)
    f_from = f1.date_input("Du", value=default_f_from, key="fab_list_from")        # REF:FAB_FROM
    f_to   = f2.date_input("Au", value=default_f_to,   key="fab_list_to")          # REF:FAB_TO
    prod_pick = f3.multiselect("Produit", ["GMQ ONE","GMQ LIVE"],
                               default=["GMQ ONE","GMQ LIVE"], key="fab_list_prod")  # REF:FAB_PROD
    
    f4, f5 = st.columns(2)
    status_opts  = ["(Tous)"] + fetch_df(
        "SELECT DISTINCT status FROM fabrications WHERE status IS NOT NULL ORDER BY 1"
    )["status"].astype(str).tolist()
    status_pick  = f4.selectbox("Statut", status_opts, index=0, key="fab_list_status")  # REF:FAB_STATUS
    client_filter = f5.text_input("Client contient", "", key="fab_list_client")         # REF:FAB_CLIENT


    fab_view = get_fabrications_filtered(f_from, f_to, prod_pick, status_pick if status_pick != "(Tous)" else None, client_filter)
    st.dataframe(fab_view, use_container_width=True)

# ---- STOCK
with tab_stock:
    st.header("Stock") #titre

    st.subheader("Ajouter un article")
    with st.form("stock_add"):
        c1, c2, c3 = st.columns(3)
        sku_new = c1.text_input("SKU *", "")
        name_new = c2.text_input("Nom *", "")
        unit_new = c3.text_input("Unité", value="pcs")
        c4, c5, c6 = st.columns(3)
        cat_new = c4.text_input("Catégorie", value="Component")
        rop_new = c5.number_input("ReorderPoint", min_value=0.0, step=1.0, value=0.0)
        qty_new = c6.number_input("QtyOnHand (initiale)", min_value=0.0, step=1.0, value=0.0)
        desc_new = st.text_input("Description", "")
        btn_add_stock = st.form_submit_button("Ajouter")

        if btn_add_stock:
            if not sku_new.strip() or not name_new.strip():
                st.error("SKU et Nom sont obligatoires.")
            else:
                try:
                    add_stock_item(sku_new.strip(), name_new.strip(), unit_new.strip(),
                                   cat_new.strip(), float(rop_new), float(qty_new), (desc_new or None))
                    st.success(f"Article {sku_new} ajouté.")
                    st.toast("Article ajouté")
                except Exception as e:
                    st.error(str(e))

    st.divider()

    st.subheader("Édition rapide")
    stock_df = get_stock()
    edited = st.data_editor(
        stock_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "reorder_point": st.column_config.NumberColumn(step=1.0, min_value=0.0),
            "qty_on_hand": st.column_config.NumberColumn(step=1.0, min_value=0.0),
        },
        key="stock_editor",
    )

    if st.button("💾 Enregistrer modifications du stock", key="stock_save_btn"):
        try:
            for r in edited.itertuples(index=False):
                upsert_stock_row({
                    "sku": r.sku, "name": r.name, "unit": r.unit, "category": r.category,
                    "reorder_point": float(r.reorder_point or 0),
                    "qty_on_hand": float(r.qty_on_hand or 0),
                    "description": r.description,
                })
            st.success("Stock enregistré")
            st.toast("Stock enregistré")
        except Exception as e:
            st.error(str(e))

    st.divider()

    st.subheader("Recherche & Export (stock)")
    s1, s2, s3 = st.columns([1, 1, 2])
    cats = ["(Toutes)"] + sorted([c for c in stock_df["category"].dropna().astype(str).unique().tolist()])
    cat_pick = s1.selectbox("Catégorie", cats, index=0)
    only_low = s2.checkbox("Sous seuil uniquement", value=False)
    q_stock = s3.text_input("Recherche (SKU / Nom / Description)", "")

    stock_view = stock_df.copy()
    if cat_pick != "(Toutes)":
        stock_view = stock_view[stock_view["category"].astype(str) == cat_pick]
    if only_low:
        stock_view = stock_view[stock_view["qty_on_hand"] < stock_view["reorder_point"]]
    if q_stock.strip():
        mask = (
            stock_view["sku"].astype(str).str.contains(q_stock, case=False, na=False)
            | stock_view["name"].astype(str).str.contains(q_stock, case=False, na=False)
            | stock_view["description"].astype(str).str.contains(q_stock, case=False, na=False)
        )
        stock_view = stock_view[mask]

    st.dataframe(stock_view, use_container_width=True)
    st.download_button(
        "⬇️ Télécharger (CSV)",
        data=to_csv_bytes(stock_view),
        file_name=f"stock_filtre_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="stock_export_btn",
    )


# ---- INVENTAIRE
with tab_invent:
    st.subheader("Inventaire (comptage & écarts)")
    resp_list = get_responsables() or ["Aymane", "Joslain", "Lise", "Robin"]
    responsable_inv = st.selectbox("Responsable inventaire", resp_list, index=0)
    ref_inv = st.text_input("Référence d'inventaire", value=f"INV-{datetime.now():%Y%m%d}")

    # Stock actuel (pour comparer)
    stock_df = get_stock()  # colonnes attendues: sku, qty_on_hand

    st.markdown("**Importer un CSV**")
    st.caption("Colonnes attendues : `SKU`, `Compté` (les noms proches sont auto-reconnus : sku, qte, counted, ...).")
    csv_file = st.file_uploader("Choisir un fichier CSV", type=["csv"], key="inv_csv_uploader")

    # --- Helpers lecture & normalisation CSV
    def _read_csv_smart(file) -> pd.DataFrame:
        import io
        # réinitialiser le curseur à chaque tentative
        raw = file.read()
        for sep in [";", ",", "\t", "|"]:
            try:
                return pd.read_csv(io.BytesIO(raw), sep=sep)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(raw))  # dernier essai

    def _coerce_inventory_df(df_raw: pd.DataFrame) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["SKU", "Compté"])
        df = df_raw.copy()
        lower = {c.lower().strip(): c for c in df.columns}

        # map SKU
        sku_col = None
        for c in ["sku", "code", "ref", "reference", "article", "id", "component_sku"]:
            if c in lower:
                sku_col = lower[c]; break
        # map Compté
        counted_col = None
        for c in ["compté", "compte", "counted", "count", "qte", "quantite", "quantité", "qty", "qty_counted", "qte_comptee", "qte_comptée"]:
            if c in lower:
                counted_col = lower[c]; break

        if not sku_col or not counted_col:
            if "SKU" in df.columns and "Compté" in df.columns:
                sku_col, counted_col = "SKU", "Compté"
            else:
                st.error("Colonnes non reconnues. Il faut au minimum 'SKU' et 'Compté'.")
                return pd.DataFrame(columns=["SKU", "Compté"])

        out = pd.DataFrame({
            "SKU": df[sku_col].astype(str),
            "Compté": pd.to_numeric(df[counted_col], errors="coerce").fillna(0.0),
        })
        out = out[out["SKU"].str.strip() != ""]
        return out

    # --- Lecture du CSV
    edited = None
    if csv_file is not None:
        try:
            raw = _read_csv_smart(csv_file)
            edited = _coerce_inventory_df(raw)
            st.success(f"CSV importé : {len(edited)} ligne(s).")
            st.dataframe(edited, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur de lecture du CSV : {e}")
            edited = None
    else:
        st.info("Importe un CSV pour calculer et valider les écarts.")

    # --- Actions
    c1, c2 = st.columns(2)
    calc = c1.button("Calculer les écarts", key="inv_calc")
    valider = c2.button("Valider ajustements", key="inv_valid")

    def compute_diffs(ed: pd.DataFrame) -> pd.DataFrame:
        if ed is None or ed.empty:
            return pd.DataFrame(columns=["SKU", "Systeme", "Compté", "Ecart", "Sens"])
        sm = stock_df[["sku", "qty_on_hand"]].rename(columns={"sku": "SKU", "qty_on_hand": "Systeme"})
        ed2 = ed.copy()
        ed2["SKU"] = ed2["SKU"].astype(str)
        merged = ed2.merge(sm, on="SKU", how="left").fillna({"Systeme": 0.0})
        merged["Compté"] = pd.to_numeric(merged["Compté"], errors="coerce").fillna(0.0)
        merged["Ecart"] = merged["Compté"] - merged["Systeme"]
        merged["Sens"] = np.where(merged["Ecart"] >= 0, "IN", "OUT")
        return merged[["SKU", "Systeme", "Compté", "Ecart", "Sens"]]

    if calc:
        if edited is None or edited.empty:
            st.warning("Aucune donnée importée. Charge un CSV d’abord.")
        else:
            diffs = compute_diffs(edited)
            st.markdown("#### Écarts calculés")
            st.dataframe(diffs, use_container_width=True)

    if valider:
        if edited is None or edited.empty:
            st.warning("Aucune donnée importée. Charge un CSV d’abord.")
        else:
            diffs = compute_diffs(edited)
            if diffs.empty:
                st.info("Aucune ligne à ajuster.")
            else:
                try:
                    for r in diffs.itertuples(index=False):
                        sku = r.SKU
                        ecart = float(r.Ecart)
                        if ecart == 0:
                            continue
                        move_type = "IN" if ecart > 0 else "OUT"
                        qty = abs(ecart)
                        record_movement_and_update(sku, move_type, qty, ref_inv, "INVENTAIRE", responsable_inv)
                    st.success("Ajustements d'inventaire enregistrés")
                except Exception as e:
                    st.error(str(e))


# ---- CLIENTS
with tab_clients:
    st.subheader("Clients")

    # === Ajouter un client
    st.markdown("### Ajouter un client")
    with st.form("add_client"):
        c1, c2 = st.columns(2)
        cname = c1.text_input("Nom du client *", "", key="cli_name")
        ctype = c2.selectbox("Type", ["Régulier", "Passage"], key="cli_type")

        c3, c4, c5 = st.columns(3)
        cphone = c3.text_input("Téléphone", "", key="cli_phone")
        cemail = c4.text_input("Email", "", key="cli_email")
        cnotes = c5.text_input("Notes", "", key="cli_notes")

        btn_cli = st.form_submit_button("Ajouter")

        if btn_cli:
            if not cname.strip():
                st.error("Le nom du client est obligatoire.")
            else:
                try:
                    _ = insert_client_return_id(
                        cname.strip(),
                        ctype,
                        cphone.strip() or None,
                        cemail.strip() or None,
                        cnotes.strip() or None,
                    )
                    st.success(f"Client « {cname.strip()} » ajouté.")
                    try:
                        st.toast("Client ajouté")  # ok si Streamlit récent
                    except Exception:
                        pass
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    # === Rechercher / Supprimer
    st.markdown("### Rechercher / Supprimer des clients")

    cq = st.text_input("Recherche client (nom, téléphone, email)", "", key="clients_search")
    cl = get_clients()  # Doit renvoyer un DataFrame avec colonnes: id, client_name, phone, email

    if cq.strip():
        # recherche en mode texte simple, pas de regex
        m = (
            cl["client_name"].astype(str).str.contains(cq, case=False, na=False, regex=False)
            | cl["phone"].astype(str).str.contains(cq, case=False, na=False, regex=False)
            | cl["email"].astype(str).str.contains(cq, case=False, na=False, regex=False)
        )
        cl = cl[m]

    st.dataframe(cl, use_container_width=True)

    # Sélection multiple basée sur les UUID, affichage "uuid – nom"
    options = cl["id"].astype(str).tolist()
    id_to_name = dict(zip(cl["id"].astype(str), cl["client_name"].astype(str)))
    del_ids = st.multiselect(
        "Sélectionne les clients à supprimer",
        options=options,
        format_func=lambda cid: f"{cid} – {id_to_name.get(cid, cid)}",
        key="cli_del_ids",
    )

    if st.button("🗑️ Supprimer la sélection", key="cli_del_btn"):
        if not del_ids:
            st.info("Aucun client sélectionné.")
        else:
            # Vérifier les références dans les OF
            used_ids_df = fetch_df(
                "SELECT DISTINCT client_id FROM fabrications WHERE client_id IS NOT NULL"
            )
            used_set = set(
                used_ids_df["client_id"].dropna().astype(str).tolist()
            ) if not used_ids_df.empty else set()

            referenced = [cid for cid in del_ids if cid in used_set]
            deletable = [cid for cid in del_ids if cid not in used_set]

            if referenced:
                names = [id_to_name.get(cid, cid) for cid in referenced]
                st.warning(
                    "Ces clients sont référencés par des ordres de fabrication et ne seront pas supprimés : "
                    + ", ".join(names)
                )

            try:
                n = delete_clients(deletable) if deletable else 0
                st.success(f"Suppression effectuée ({n} client(s)).")
                try:
                    st.toast("Suppression terminée")
                except Exception:
                    pass
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de la suppression : {e}")
    


# ============================ HELPERS BOM ============================

# Helper pour lister les colonnes d'une table (PostgreSQL / SQLite compatibles)
def get_stock_components() -> pd.DataFrame:
    """Colonnes normalisées pour l'UI: id (=sku), item_name (=name), unit."""
    sql = """
        SELECT
            sku::text   AS id,
            name::text  AS item_name,
            COALESCE(unit::text, '') AS unit
        FROM stock
        ORDER BY name ASC
    """
    return fetch_df(sql)

def get_bom_full(table_name: str) -> pd.DataFrame:
    """
    Charge TOUTE la table BOM (bom_gmq_one | bom_gmq_live) avec les libellés depuis stock.
    Colonnes renvoyées: component_sku, item_name, unit, qty_per_unit, description
    """
    if table_name not in {"bom_gmq_one", "bom_gmq_live"}:
        raise ValueError("Table BOM inconnue")

    sql = f"""
        SELECT
            b.component_sku::text              AS component_sku,
            COALESCE(s.name, '??')            AS item_name,
            COALESCE(s.unit, '')              AS unit,
            COALESCE(b.qty_per_unit, 1)       AS qty_per_unit,
            COALESCE(b.description, '')       AS description
        FROM {table_name} b
        LEFT JOIN stock s ON s.sku = b.component_sku
        ORDER BY item_name
    """
    return fetch_df(sql)

def save_bom_full_replace(table_name: str, df: pd.DataFrame, stock_df: pd.DataFrame) -> int:
    """
    Remplace TOUTE la table BOM par le contenu de df.
    - Ne garde que les lignes avec component_sku ∈ stock.sku et qty_per_unit > 0.
    - Retourne le nombre de lignes insérées.
    """
    if table_name not in {"bom_gmq_one", "bom_gmq_live"}:
        raise ValueError("Table BOM inconnue")

    if df is None or df.empty:
        # Sécurité: on ne vide pas la table si l'éditeur est vide
        return 0

    valid_skus = set(stock_df["id"].astype(str))

    work = df.copy()
    # Colonnes minimales
    for c in ["component_sku", "qty_per_unit", "description"]:
        if c not in work.columns:
            work[c] = "" if c != "qty_per_unit" else 0

    work["component_sku"] = work["component_sku"].astype(str).str.strip()
    work = work[work["component_sku"].isin(valid_skus)]
    work = work[pd.to_numeric(work["qty_per_unit"], errors="coerce").fillna(0) > 0]

    rows = [
        {
            "component_sku": str(r["component_sku"]),
            "qty_per_unit": float(r["qty_per_unit"]),
            "description": str(r.get("description", "") or "")
        }
        for _, r in work.iterrows()
    ]

    with engine.begin() as conn:
        # Remplacement complet
        conn.execute(text(f"DELETE FROM {table_name}"))
        if rows:
            stmt = text(f"""
                INSERT INTO {table_name} (component_sku, qty_per_unit, description)
                VALUES (:component_sku, :qty_per_unit, :description)
            """)
            conn.execute(stmt, rows)
    return len(rows)


def _load_bom_full_into_state(table_name: str):
    state_key = f"bom_full_df_{table_name}"
    st.session_state[state_key] = get_bom_full(table_name)

# =============================== ONGLET BOM ===============================
with tab_bom:
    st.subheader("BOM — GMQ (édition par table)")

    table_choice = st.radio(
        "Table BOM à modifier",
        options=["bom_gmq_one", "bom_gmq_live"],
        horizontal=True,
        key="bom_table_choice",
    )

    # Référentiel composants
    stock_df = get_stock_components()
    stock_id_to_name = dict(zip(stock_df["id"].astype(str), stock_df["item_name"]))
    stock_id_to_unit = dict(zip(stock_df["id"].astype(str), stock_df["unit"]))

    # --------- CHARGEMENT CONTRÔLÉ (ne pas écraser à chaque run !) ---------
    state_key = f"bom_full_df_{table_choice}"
    # mémoriser la dernière table
    if "bom_last_table" not in st.session_state:
        st.session_state["bom_last_table"] = table_choice

    # 1) première fois pour cette table
    if state_key not in st.session_state:
        _load_bom_full_into_state(table_choice)

    # 2) si l’utilisateur change de table → recharger la nouvelle, ne pas toucher sinon
    if st.session_state["bom_last_table"] != table_choice:
        _load_bom_full_into_state(table_choice)
        st.session_state["bom_last_table"] = table_choice

    # 3) bouton de refresh manuel (pas automatique à chaque run)
    if st.button("🔄 Recharger depuis la base", key="bom_refresh_btn"):
        _load_bom_full_into_state(table_choice)

    # ----------------------------------------------------------------------

    st.markdown("### Ajouter des composants (depuis le stock)")
    with st.expander("➕ Ajouter"):
        added_skus = st.multiselect(
            "Composants à ajouter (SKU)",
            options=stock_df["id"].astype(str).tolist(),
            format_func=lambda sku: f"{stock_id_to_name.get(sku, '??')} — {sku}",
            key="bom_add_skus",
        )
        default_qty = st.number_input("Quantité par défaut", min_value=0.0, value=1.0, step=1.0, key="bom_default_qty_all")
        if st.button("Ajouter à la table courante", key="bom_add_btn_all"):
            current = st.session_state[state_key].copy()
            existing = set(current["component_sku"].astype(str)) if not current.empty else set()
            to_add = [sku for sku in added_skus if sku not in existing]
            if to_add:
                add_rows = pd.DataFrame({
                    "component_sku": to_add,
                    "item_name": [stock_id_to_name.get(sku, "??") for sku in to_add],
                    "unit": [stock_id_to_unit.get(sku, "") for sku in to_add],
                    "qty_per_unit": [default_qty for _ in to_add],
                    "description": ["" for _ in to_add],
                })
                st.session_state[state_key] = pd.concat([current, add_rows], ignore_index=True)
                st.success(f"{len(to_add)} composant(s) ajouté(s).")
            else:
                st.info("Aucun nouveau composant à ajouter.")

    st.markdown("### Éditer la BOM")
    edited_df = st.data_editor(
        st.session_state[state_key],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "component_sku": st.column_config.TextColumn("SKU composant"),
            "item_name": st.column_config.TextColumn("Nom composant", disabled=True),
            "unit": st.column_config.TextColumn("Unité", disabled=True),
            "qty_per_unit": st.column_config.NumberColumn("Quantité par unité", min_value=0.0, step=0.1),
            "description": st.column_config.TextColumn("Description"),
        },
        key=f"bom_editor_full_{table_choice}",
    )

    c1, c2, c3 = st.columns(3)
    if c1.button("🧹 Vider (local)", key="bom_clear_full"):
        st.session_state[state_key] = pd.DataFrame(columns=["component_sku", "item_name", "unit", "qty_per_unit", "description"])
        st.info("Table locale vidée — non enregistrée.")

    if c2.button("🗑️ Astuce suppression (local)", key="bom_hint_del_full"):
        st.info("Pour supprimer une ligne, mets qty_per_unit à 0 puis Enregistrer (les lignes qty=0 seront ignorées).")

    if c3.button("💾 Enregistrer dans la base", key="bom_save_full"):
        try:
            # IMPORTANT : on sauve **le DataFrame édité**, pas le rechargé
            n = save_bom_full_replace(table_choice, edited_df, stock_df)
            if n == 0 and (edited_df is None or edited_df.empty):
                st.warning("Éditeur vide → par sécurité, la table n’a pas été modifiée.")
            else:
                st.success(f"Table {table_choice} enregistrée ({n} ligne(s)).")
                try:
                    st.toast("BOM enregistrée")
                except Exception:
                    pass
                # maintenant on recharge depuis la DB (après save), puis on rerun
                _load_bom_full_into_state(table_choice)
                st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l’enregistrement : {e}")


