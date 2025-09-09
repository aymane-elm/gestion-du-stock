import os
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Result

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Stock & Fabrication (PostgreSQL)", layout="wide")
st.title("Gestion de stock & fabrication – SQL (PostgreSQL)")

DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_gW2a0Hlfzpxn@ep-divine-scene-agixk2f3-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# =========================
# SQL – Connexion & DDL
# =========================
@st.cache_resource
def get_engine() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True)

engine = get_engine()



def run_ddl():
    with engine.begin() as conn:
        # Seed de base responsables si vide
        r = conn.execute(text("SELECT COUNT(*) FROM responsables")).scalar_one()
        if r == 0:
            conn.execute(
                text("INSERT INTO responsables(name) VALUES (:a),(:b),(:c),(:d)"),
                {"a": "Aymane", "b": "Joslain", "c": "Lise", "d": "Robin"},
            )

run_ddl()

# =========================
# HELPERS SQL
# =========================
def fetch_df(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params)

def execute(sql: str, params: Dict[str, Any] | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def executemany(sql: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(text(sql), rows)

# =========================
# LOGIQUE MÉTIER
# =========================
def get_responsables() -> List[str]:
    df = fetch_df("SELECT name FROM responsables ORDER BY name")
    return df["name"].astype(str).tolist()

def get_clients() -> pd.DataFrame:
    return fetch_df("SELECT * FROM clients ORDER BY client_name")

def get_stock() -> pd.DataFrame:
    df = fetch_df("""
        SELECT sku, name, unit, category,
               COALESCE(reorder_point,0) AS reorder_point,
               COALESCE(qty_on_hand,0) AS qty_on_hand,
               description
        FROM stock
        ORDER BY sku
    """)
    # Types propres
    df["reorder_point"] = pd.to_numeric(df["reorder_point"], errors="coerce").fillna(0.0)
    df["qty_on_hand"]   = pd.to_numeric(df["qty_on_hand"], errors="coerce").fillna(0.0)
    return df

def upsert_stock_row(row: Dict[str, Any]) -> None:
    execute("""
        INSERT INTO stock (sku, name, unit, category, reorder_point, qty_on_hand, description)
        VALUES (:sku, :name, :unit, :category, :reorder_point, :qty_on_hand, :description)
        ON CONFLICT (sku) DO UPDATE SET
          name = EXCLUDED.name,
          unit = EXCLUDED.unit,
          category = EXCLUDED.category,
          reorder_point = EXCLUDED.reorder_point,
          qty_on_hand = EXCLUDED.qty_on_hand,
          description = EXCLUDED.description
    """, row)

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
                               mo_id: str | None = None) -> float:
    """
    Transaction : met à jour le stock et ajoute une ligne mouvement.
    Retourne la nouvelle quantité on hand.
    """
    if qty < 0:
        raise ValueError("La quantité doit être >= 0")
    delta = qty if move_type == "IN" else -qty
    with engine.begin() as conn:
        # Vérifier existence SKU
        r = conn.execute(text("SELECT qty_on_hand FROM stock WHERE sku=:sku FOR UPDATE"), {"sku": sku}).first()
        if r is None:
            raise ValueError(f"SKU introuvable: {sku}")
        current = float(r[0] or 0)
        new_val = current + delta
        conn.execute(text("UPDATE stock SET qty_on_hand=:q WHERE sku=:sku"), {"q": new_val, "sku": sku})
        conn.execute(text("""
            INSERT INTO mouvements(date, sku, type, qty, ref, location, mo_id, responsable)
            VALUES (now(), :sku, :type, :qty, :ref, :loc, :mo, :resp)
        """), {"sku": sku, "type": move_type, "qty": float(qty), "ref": ref, "loc": location, "mo": mo_id, "resp": responsable})
        return new_val

def get_mouvements_filtered(d_from: date, d_to: date, types: List[str],
                            sku_like: str | None, responsable: str | None) -> pd.DataFrame:
    q = """
        SELECT id, date, sku, type, qty, ref, location, mo_id, responsable
        FROM mouvements
        WHERE date::date BETWEEN :dfrom AND :dto
    """
    params: Dict[str, Any] = {"dfrom": d_from, "dto": d_to}
    if types:
        q += " AND type = ANY(:types)"
        params["types"] = types
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
    q = """
        SELECT mo_id, date, due_date, product, qty, status, ref, responsable, client
        FROM fabrications
        WHERE date::date BETWEEN :dfrom AND :dto
    """
    params: Dict[str, Any] = {"dfrom": d_from, "dto": d_to}
    if products:
        q += " AND product = ANY(:prods)"
        params["prods"] = products
    if status and status != "(Tous)":
        q += " AND status = :st"
        params["st"] = status
    if client_like and client_like.strip():
        q += " AND client ILIKE :cl"
        params["cl"] = f"%{client_like.strip()}%"
    q += " ORDER BY COALESCE(due_date, CURRENT_DATE) ASC, date DESC"
    return fetch_df(q, params)

def get_bom(product: str) -> pd.DataFrame:
    if product == "GMQ ONE":
        return fetch_df("""
            SELECT component_sku AS componentsku, qty_per_unit AS qtyperunit, description
            FROM bom_gmq_one
        """)
    elif product == "GMQ LIVE":
        return fetch_df("""
            SELECT component_sku AS componentsku, qty_per_unit AS qtyperunit, description
            FROM bom_gmq_live
        """)
    else:
        raise ValueError("Produit inconnu")

def check_availability_sql(product: str, qty_make: float) -> Tuple[pd.DataFrame, bool]:
    bom = get_bom(product)
    stock = get_stock()[["sku","qty_on_hand"]].rename(columns={"sku":"componentsku"})
    df = bom.merge(stock, on="componentsku", how="left")
    df["qtyperunit"] = pd.to_numeric(df["qtyperunit"], errors="coerce").fillna(0.0)
    df["qty_on_hand"] = pd.to_numeric(df["qty_on_hand"], errors="coerce").fillna(0.0)
    df["Besoin (total)"] = df["qtyperunit"] * float(qty_make or 0)
    df["Stock dispo"] = df["qty_on_hand"]
    df["Manque"] = (df["Besoin (total)"] - df["Stock dispo"]).clip(lower=0.0)
    out = df.rename(columns={
        "componentsku": "ComponentSKU",
        "qtyperunit": "QtyPerUnit",
        "description": "Description"
    })[["ComponentSKU","QtyPerUnit","Besoin (total)","Stock dispo","Manque","Description"]]
    out = out.sort_values(["Manque","ComponentSKU"], ascending=[False, True]).reset_index(drop=True)
    ok = float(out["Manque"].sum()) == 0.0
    return out, ok

def post_fabrication(product: str, qty_make: float, due_date: date,
                     ref: str, responsable: str, client: str | None) -> str:
    """
    Poste un OF :
      - crée row dans fabrications
      - sort les composants (mouvements OUT)
      - entre le produit fini (IN)
      - le tout dans une transaction
    """
    req_df, ok = check_availability_sql(product, qty_make)
    if not ok:
        missing = req_df[req_df["Manque"] > 0]
        raise ValueError("Stock insuffisant pour poster l'OF")

    mo_id = uuid.uuid4().hex[:8].upper()
    fin_sku = "GMQ-ONE" if product == "GMQ ONE" else "GMQ-LIVE"

    with engine.begin() as conn:
        # 1) Insert fabrication
        conn.execute(text("""
            INSERT INTO fabrications(mo_id, date, due_date, product, qty, status, ref, responsable, client)
            VALUES (:mo, now(), :due, :prod, :qty, 'Posté', :ref, :resp, :client)
        """), {"mo": mo_id, "due": due_date, "prod": product, "qty": float(qty_make),
               "ref": ref, "resp": responsable, "client": client})

        # 2) Composants OUT
        for r in req_df.itertuples(index=False):
            comp_sku = r.ComponentSKU
            need = float(r._2) if hasattr(r, "_2") else float(getattr(r, "Besoin (total)"))
            # MAJ stock composant
            conn.execute(text("UPDATE stock SET qty_on_hand = COALESCE(qty_on_hand,0) - :q WHERE sku=:s"),
                         {"q": need, "s": comp_sku})
            # Mouvement
            conn.execute(text("""
                INSERT INTO mouvements(date, sku, type, qty, ref, location, mo_id, responsable)
                VALUES (now(), :sku, 'OUT', :qty, :ref, 'PROD', :mo, :resp)
            """), {"sku": comp_sku, "qty": need, "ref": ref, "mo": mo_id, "resp": responsable})

        # 3) Produit fini IN
        conn.execute(text("UPDATE stock SET qty_on_hand = COALESCE(qty_on_hand,0) + :q WHERE sku=:s"),
                     {"q": float(qty_make), "s": fin_sku})
        conn.execute(text("""
            INSERT INTO mouvements(date, sku, type, qty, ref, location, mo_id, responsable)
            VALUES (now(), :sku, 'IN', :qty, :ref, 'STOCK', :mo, :resp)
        """), {"sku": fin_sku, "qty": float(qty_make), "ref": ref, "mo": mo_id, "resp": responsable})

    return mo_id

def add_client(name: str, ctype: str, phone: str | None, email: str | None, notes: str | None):
    execute("""
        INSERT INTO clients (client_id, client_name, type, phone, email, notes)
        VALUES (:id, :name, :type, :phone, :email, :notes)
    """, {"id": "C-" + uuid.uuid4().hex[:8].upper(), "name": name.strip(),
          "type": ctype, "phone": phone or None, "email": email or None, "notes": notes or None})

def delete_clients(ids: List[str]):
    if not ids:
        return
    with engine.begin() as conn:
        # avertissement côté UI déjà géré ; ici suppression directe
        conn.execute(text("DELETE FROM clients WHERE client_id = ANY(:ids)"), {"ids": ids})

# =========================
# UTILS EXPORT
# =========================
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# =========================
# UI – TABS
# =========================
tab_dash, tab_moves, tab_mo, tab_stock, tab_compos, tab_invent, tab_clients, tab_export = st.tabs([
    "Dashboard","Mouvements","Ordres de fabrication","Stock",
    "Composants","Inventaire","Clients","Export CSV"
])

# ---- DASHBOARD
with tab_dash:
    st.subheader("Aperçu")
    stock_df = get_stock()
    total_skus = int(len(stock_df))
    total_qty = float(stock_df["qty_on_hand"].sum())
    mo_posted = int(fetch_df("SELECT COUNT(*) AS c FROM fabrications WHERE status='Posté'")["c"].iat[0])
    low = stock_df[ stock_df["qty_on_hand"] < stock_df["reorder_point"] ]
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
    resp_list = get_responsables() or ["Aymane","Joslain","Lise","Robin"]
    stock_df = get_stock()

    st.subheader("Ajouter un mouvement")
    with st.form("mv_form"):
        col_a, col_b = st.columns(2)
        sku = col_a.selectbox("SKU", stock_df["sku"].astype(str).tolist())
        responsable = col_b.selectbox("Responsable", resp_list, index=0)
        move_type = st.radio("Type", ["IN","OUT"], horizontal=True)
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
    # bornes par défaut
    mv_all = fetch_df("SELECT MIN(date)::date AS dmin, MAX(date)::date AS dmax FROM mouvements")
    default_from = (mv_all["dmin"].iat[0] or (date.today() - timedelta(days=30)))
    default_to   = (mv_all["dmax"].iat[0] or date.today())

    c1, c2, c3 = st.columns(3)
    d_from = c1.date_input("Du", value=default_from)
    d_to   = c2.date_input("Au", value=default_to)
    types = c3.multiselect("Type", options=["IN","OUT"], default=["IN","OUT"])

    c4, c5 = st.columns(2)
    sku_filter = c4.text_input("Filtre SKU (contient)", "")
    resp_opts = ["(Tous)"] + resp_list
    resp_pick = c5.selectbox("Responsable", resp_opts, index=0)

    mv_view = get_mouvements_filtered(d_from, d_to, types, sku_filter, resp_pick if resp_pick!="(Tous)" else None)
    st.dataframe(mv_view, use_container_width=True)

# ---- ORDRES DE FABRICATION (vérif + création + filtres + tableau)
with tab_mo:
    st.header("Ordres de fabrication")
    resp_list = get_responsables() or ["Aymane","Joslain","Lise","Robin"]
    clients_df = get_clients()
    clients_list = clients_df["client_name"].astype(str).tolist() if not clients_df.empty else []

    st.subheader("Créer un OF")
    with st.form("mo_form"):
        col1, col2 = st.columns(2)
        product = col1.selectbox("Produit fini", ["GMQ ONE","GMQ LIVE"])
        responsable = col2.selectbox("Responsable", resp_list, index=0)
        col3, col4, col5 = st.columns([1,1,2])
        qty_make = col3.number_input("Quantité à produire", min_value=0.0, step=1.0)
        due_date = col4.date_input("Date d'échéance", value=date.today() + timedelta(days=7))
        ref = col5.text_input("Référence OF", value="OF-AUTO")

        st.markdown("**Client associé à l'OF**")
        colc1, colc2 = st.columns(2)
        client_pick = ["(aucun)"] + clients_list + ["Client de passage (saisie)"]
        client_choice = colc1.selectbox("Client", client_pick, index=0)
        client_free = None
        if client_choice == "Client de passage (saisie)":
            client_free = colc2.text_input("Nom du client (passage)", value="")

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
                    manques = req_df.loc[req_df["Manque"] > 0, ["ComponentSKU","Manque"]]
                    manques_str = ", ".join([f"{r.ComponentSKU} (-{r.Manque:.0f})" if float(r.Manque).is_integer() else f"{r.ComponentSKU} (-{r.Manque})" for r in manques.itertuples()])
                    st.error(f"Stock insuffisant. Composants manquants : {manques_str}")
                else:
                    st.success("Stock OK pour l'OF.")

                if post_clicked and ok:
                    client_final = None
                    if client_choice == "Client de passage (saisie)":
                        client_final = (client_free or "").strip() or None
                    elif client_choice not in ["(aucun)", "Client de passage (saisie)"]:
                        client_final = client_choice
                    try:
                        mo_id = post_fabrication(product, qty_make, due_date, ref, responsable, client_final)
                        st.success(f"OF {mo_id} posté par {responsable} (échéance {due_date:%Y-%m-%d}).")
                        st.toast("OF posté")
                    except Exception as e:
                        st.error(str(e))

    st.divider()
    st.subheader("Liste des ordres de fabrication")
    fab_mm = fetch_df("SELECT MIN(date)::date AS dmin, MAX(date)::date AS dmax FROM fabrications")
    default_f_from = (fab_mm["dmin"].iat[0] or (date.today() - timedelta(days=30)))
    default_f_to   = (fab_mm["dmax"].iat[0] or date.today())

    f1, f2, f3 = st.columns(3)
    f_from = f1.date_input("Du", value=default_f_from)
    f_to   = f2.date_input("Au", value=default_f_to)
    prod_pick = f3.multiselect("Produit", ["GMQ ONE","GMQ LIVE"], default=["GMQ ONE","GMQ LIVE"])

    f4, f5 = st.columns(2)
    status_opts = ["(Tous)"] + fetch_df("SELECT DISTINCT status FROM fabrications WHERE status IS NOT NULL ORDER BY 1")["status"].astype(str).tolist()
    status_pick = f4.selectbox("Statut", status_opts, index=0)
    client_filter = f5.text_input("Client contient", "")

    fab_view = get_fabrications_filtered(f_from, f_to, prod_pick, status_pick if status_pick!="(Tous)" else None, client_filter)
    st.dataframe(fab_view, use_container_width=True)

# ---- STOCK (ajout + édition rapide + filtres + export)
with tab_stock:
    st.header("Stock")

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
        key="stock_editor"
    )

    if st.button("💾 Enregistrer modifications du stock", key="stock_save_btn"):
        try:
            # upsert ligne par ligne (sécurisé et simple)
            for r in edited.itertuples(index=False):
                upsert_stock_row({
                    "sku": r.sku, "name": r.name, "unit": r.unit, "category": r.category,
                    "reorder_point": float(r.reorder_point or 0),
                    "qty_on_hand": float(r.qty_on_hand or 0),
                    "description": r.description
                })
            st.success("Stock enregistré")
            st.toast("Stock enregistré")
        except Exception as e:
            st.error(str(e))

    st.divider()

    st.subheader("Recherche & Export (stock)")
    s1, s2, s3 = st.columns([1,1,2])
    cats = ["(Toutes)"] + sorted([c for c in stock_df["category"].dropna().astype(str).unique().tolist()])
    cat_pick = s1.selectbox("Catégorie", cats, index=0)
    only_low = s2.checkbox("Sous seuil uniquement", value=False)
    q_stock = s3.text_input("Recherche (SKU / Nom / Description)", "")

    stock_view = stock_df.copy()
    if cat_pick != "(Toutes)":
        stock_view = stock_view[stock_view["category"].astype(str) == cat_pick]
    if only_low:
        stock_view = stock_view[ stock_view["qty_on_hand"] < stock_view["reorder_point"] ]
    if q_stock.strip():
        mask = (
            stock_view["sku"].astype(str).str.contains(q_stock, case=False, na=False) |
            stock_view["name"].astype(str).str.contains(q_stock, case=False, na=False) |
            stock_view["description"].astype(str).str.contains(q_stock, case=False, na=False)
        )
        stock_view = stock_view[mask]

    st.dataframe(stock_view, use_container_width=True)
    st.download_button(
        "⬇️ Télécharger (CSV)",
        data=to_csv_bytes(stock_view),
        file_name=f"stock_filtre_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="stock_export_btn"
    )

# ---- COMPOSANTS : vue ciblée + ajout
with tab_compos:
    st.subheader("Recherche de composants")
    q = st.text_input("Recherche (SKU / Nom / Description)", "", key="comp_q")
    comp_df = fetch_df("""
        SELECT s.*
        FROM stock s
        WHERE LOWER(COALESCE(s.category,'')) = 'component'
        ORDER BY s.sku
    """)
    if q.strip():
        mask = (
            comp_df["sku"].astype(str).str.contains(q, case=False, na=False) |
            comp_df["name"].astype(str).str.contains(q, case=False, na=False) |
            comp_df["description"].astype(str).str.contains(q, case=False, na=False)
        )
        comp_df = comp_df[mask]
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("### Ajouter un composant")
    with st.form("add_component"):
        c1, c2, c3 = st.columns(3)
        sku_new = c1.text_input("SKU *", "", key="comp_sku_new")
        name_new = c2.text_input("Nom *", "", key="comp_name_new")
        unit_new = c3.text_input("Unité", value="pcs", key="comp_unit_new")
        c4, c5, c6 = st.columns(3)
        cat_new = c4.text_input("Catégorie", value="Component", key="comp_cat_new")
        rop_new = c5.number_input("ReorderPoint", min_value=0.0, step=1.0, value=0.0, key="comp_rop_new")
        qty_new = c6.number_input("QtyOnHand (initiale)", min_value=0.0, step=1.0, value=0.0, key="comp_qty_new")
        desc_new = st.text_input("Description", "", key="comp_desc_new")
        btn_add = st.form_submit_button("Ajouter")

        if btn_add:
            if not sku_new.strip() or not name_new.strip():
                st.error("SKU et Nom sont obligatoires.")
            else:
                try:
                    add_stock_item(sku_new.strip(), name_new.strip(), unit_new.strip(),
                                   cat_new.strip(), float(rop_new), float(qty_new), (desc_new or None))
                    st.success(f"Composant {sku_new} ajouté.")
                except Exception as e:
                    st.error(str(e))

# ---- INVENTAIRE : comptage + écarts
with tab_invent:
    st.subheader("Inventaire (comptage & écarts)")
    resp_list = get_responsables() or ["Aymane","Joslain","Lise","Robin"]
    responsable_inv = st.selectbox("Responsable inventaire", resp_list, index=0)
    ref_inv = st.text_input("Référence d'inventaire", value=f"INV-{datetime.now():%Y%m%d}")

    st.markdown("**Saisir le comptage**")
    stock_df = get_stock()
    template = pd.DataFrame({"SKU": [], "Compté": []})
    edited = st.data_editor(
        template,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "SKU": st.column_config.SelectboxColumn(options=stock_df["sku"].astype(str).tolist()),
            "Compté": st.column_config.NumberColumn(min_value=0.0, step=1.0)
        },
        key="inv_editor"
    )

    c1, c2 = st.columns(2)
    calc = c1.button("Calculer les écarts", key="inv_calc")
    valider = c2.button("Valider ajustements", key="inv_valid")

    def compute_diffs(ed: pd.DataFrame) -> pd.DataFrame:
        if ed is None or ed.empty:
            return pd.DataFrame(columns=["SKU","Systeme","Compté","Ecart","Sens"])
        sm = stock_df[["sku","qty_on_hand"]].rename(columns={"sku":"SKU","qty_on_hand":"Systeme"})
        ed2 = ed.copy()
        ed2["SKU"] = ed2["SKU"].astype(str)
        merged = ed2.merge(sm, on="SKU", how="left").fillna({"Systeme": 0.0})
        merged["Compté"] = pd.to_numeric(merged["Compté"], errors="coerce").fillna(0.0)
        merged["Ecart"] = merged["Compté"] - merged["Systeme"]
        merged["Sens"] = np.where(merged["Ecart"]>=0, "IN", "OUT")
        return merged[["SKU","Systeme","Compté","Ecart","Sens"]]

    if calc:
        diffs = compute_diffs(edited)
        st.markdown("#### Écarts calculés")
        st.dataframe(diffs, use_container_width=True)

    if valider:
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

# ---- CLIENTS : ajout + suppression
with tab_clients:
    st.subheader("Clients")

    st.markdown("### Ajouter un client")
    with st.form("add_client"):
        c1, c2 = st.columns(2)
        cname = c1.text_input("Nom du client *", "", key="cli_name")
        ctype = c2.selectbox("Type", ["Régulier","Passage"], key="cli_type")
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
                    add_client(cname.strip(), ctype, cphone.strip() or None, cemail.strip() or None, cnotes.strip() or None)
                    st.success(f"Client « {cname.strip()} » ajouté.")
                    st.toast("Client ajouté")
                except Exception as e:
                    st.error(str(e))

    st.markdown("### Rechercher / Supprimer des clients")
    cq = st.text_input("Recherche client (nom, téléphone, email)", "", key="clients_search")
    cl = get_clients()
    if cq.strip():
        m = (
            cl["client_name"].astype(str).str.contains(cq, case=False, na=False) |
            cl["phone"].astype(str).str.contains(cq, case=False, na=False) |
            cl["email"].astype(str).str.contains(cq, case=False, na=False)
        )
        cl = cl[m]

    st.dataframe(cl, use_container_width=True)
    del_ids = st.multiselect(
        "Sélectionne les clients à supprimer",
        options=cl["client_id"].astype(str).tolist(),
        format_func=lambda cid: f"{cid} – {cl.loc[cl['client_id']==cid,'client_name'].values[0] if (cl['client_id']==cid).any() else cid}",
        key="cli_del_ids"
    )
    if st.button("🗑️ Supprimer la sélection", key="cli_del_btn"):
        if not del_ids:
            st.info("Aucun client sélectionné.")
        else:
            # avertissement si référencés par des OF (comparaison par nom, comme la colonne fabrications.client stocke le nom)
            used_names = fetch_df("SELECT DISTINCT client FROM fabrications WHERE client IS NOT NULL")["client"].dropna().astype(str).tolist()
            names_to_del = get_clients().set_index("client_id").loc[del_ids, "client_name"].astype(str).tolist()
            referenced = [n for n in names_to_del if n in set(used_names)]
            if referenced:
                st.warning("Attention : des ordres de fabrication référencent ces clients : " + ", ".join(referenced))
            delete_clients(del_ids)
            st.success(f"Suppression effectuée ({len(del_ids)} client(s)).")

# ---- EXPORT CSV (global)
with tab_export:
    st.subheader("Exports CSV")

    # Stock
    st.markdown("### Export Stock")
    stock_df = get_stock()
    st.download_button(
        "⬇️ Télécharger Stock (CSV)",
        data=to_csv_bytes(stock_df),
        file_name=f"stock_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="exp_stock_btn"
    )

    st.divider()

    # Mouvements (avec filtres)
    st.markdown("### Export Mouvements")
    mv_all = fetch_df("SELECT MIN(date)::date AS dmin, MAX(date)::date AS dmax FROM mouvements")
    default_from = (mv_all["dmin"].iat[0] or (date.today() - timedelta(days=30)))
    default_to   = (mv_all["dmax"].iat[0] or date.today())
    c1, c2, c3 = st.columns(3)
    d_from = c1.date_input("Du", value=default_from, key="exp_mv_from")
    d_to   = c2.date_input("Au", value=default_to, key="exp_mv_to")
    types = c3.multiselect("Type", options=["IN","OUT"], default=["IN","OUT"], key="exp_mv_types")
    c4, c5 = st.columns(2)
    sku_filter = c4.text_input("Filtre SKU", "", key="exp_mv_sku")
    resp_opts = ["(Tous)"] + get_responsables()
    resp_pick = c5.selectbox("Responsable", resp_opts, index=0, key="exp_mv_resp")

    mv_exp = get_mouvements_filtered(d_from, d_to, types, sku_filter, resp_pick if resp_pick!="(Tous)" else None)
    st.download_button(
        "⬇️ Télécharger Mouvements filtrés (CSV)",
        data=to_csv_bytes(mv_exp),
        file_name=f"mouvements_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="exp_mv_btn"
    )

    st.divider()

    # Fabrications (avec filtres)
    st.markdown("### Export Fabrications")
    fab_mm = fetch_df("SELECT MIN(date)::date AS dmin, MAX(date)::date AS dmax FROM fabrications")
    default_f_from = (fab_mm["dmin"].iat[0] or (date.today() - timedelta(days=30)))
    default_f_to   = (fab_mm["dmax"].iat[0] or date.today())
    f1, f2, f3 = st.columns(3)
    f_from = f1.date_input("Du", value=default_f_from, key="exp_fab_from")
    f_to   = f2.date_input("Au", value=default_f_to, key="exp_fab_to")
    prod_pick = f3.multiselect("Produit", ["GMQ ONE","GMQ LIVE"], default=["GMQ ONE","GMQ LIVE"], key="exp_fab_prod")
    f4, f5 = st.columns(2)
    status_opts = ["(Tous)"] + fetch_df("SELECT DISTINCT status FROM fabrications WHERE status IS NOT NULL ORDER BY 1")["status"].astype(str).tolist()
    status_pick = f4.selectbox("Statut", status_opts, index=0, key="exp_fab_status")
    client_filter = f5.text_input("Client contient", "", key="exp_fab_client")

    fab_exp = get_fabrications_filtered(f_from, f_to, prod_pick, status_pick if status_pick!="(Tous)" else None, client_filter)
    st.download_button(
        "⬇️ Télécharger Fabrications filtrées (CSV)",
        data=to_csv_bytes(fab_exp),
        file_name=f"fabrications_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="exp_fab_btn"
    )
