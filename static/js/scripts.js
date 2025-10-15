function onAggregationChange(data, attr) {
    const attrsToRemove = new Set();
    data.forEach((a) =>
        a.related.forEach((r) => {
            if (r.agg) attrsToRemove.add(r.attribute);
        })
    );
    data.forEach((a) => {
        a.related = a.related.filter((r) => !attrsToRemove.has(r.attribute));
        a.split_attributes = a.split_attributes.filter((s) => !attrsToRemove.has(s.attribute));
    });

    if (!attr.aggregation) return;

    fetch('/aggregate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attribute: attr }),
    })
        .then((res) => res.json())
        .then((json) => {
            const resAttr = json.aggregated_attribute;
            data.forEach((a) =>
                a.related.forEach((r) => {
                    if (
                        r.attribute === resAttr.original_attribute &&
                        r.type === resAttr.type &&
                        r.qualifier === resAttr.qualifier
                    ) {
                        a.related.push({
                            attribute: resAttr.attribute,
                            type: 'EVENT',
                            qualifier: resAttr.qualifier,
                            vals: resAttr.vals,
                            agg: true,
                            selected: false,
                        });
                    }
                })
            );
        })
        .catch((err) => console.error('Aggregation fehlgeschlagen:', err));
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.table tbody tr').forEach((row) => {
        const selectCheckbox = row.querySelector('.form-check-input:not([x-model$="numeric"])');
        const numericCheckbox = row.querySelector('.form-check-input[x-model$="numeric"]');
        if (!selectCheckbox || !numericCheckbox) return;

        numericCheckbox.addEventListener('change', () => {
            if (numericCheckbox.checked) {
                selectCheckbox.checked = true;
                selectCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
        selectCheckbox.addEventListener('change', () => {
            if (!selectCheckbox.checked) {
                numericCheckbox.checked = false;
                numericCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    });
});

document.addEventListener('alpine:init', () => {
    Alpine.data('multiselect', () => ({
        init() {
            this.$nextTick(() => {});
        },
    }));

    Alpine.data('discretizationForm', () => ({
        defaultParameters: {
            'equal-freq': { bins: 5 },
            'equal-width': { bins: 5 },
            'chi-merge': { labels: [], max_interval: 5, significance: 0.05 },
            'k-means': { clusters: 5, labels: [] },
        },
        data: {
            attributes: window.initialAttributes.map((attr) => ({
                attribute: attr.attribute,
                type: attr.type,
                qualifier: attr.qualifier,
                related: attr.related.map((r) => ({
                    attribute: r.attribute,
                    type: r.type,
                    qualifier: r.qualifier,
                    vals: r.vals,
                    selected: false,
                })),
                selected: false,
                numeric: false,
                aggregation: '',
                algorithm: { name: 'equal-freq', parameters: { bins: 5 } },
                split_attributes: [],
            })),
        },
        submitForm() {
            const payload = JSON.parse(JSON.stringify(this.data));
            payload.attributes.forEach((attr) => {
                delete attr.related;
                if (!attr.numeric) {
                    delete attr.algorithm;
                    delete attr.split_attributes;
                }
            });
            fetch(this.$el.action, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            })
                .then((res) => res.json())
                .then((json) => {
                    if (json.status === 'success' && json.redirect_url) {
                        window.location.href = json.redirect_url;
                    } else {
                        alert('Fehler beim Verarbeiten!');
                    }
                });
        },
    }));

    Alpine.data('itemsForm', () => ({
        defaultParameters: {
            itemset: { min_sup: 0.1 },
            associationrule: { min_sup: 0.02, min_conf: 0.5, min_lift: 1.0 },
            classificationrule: {
                min_sup: 0.02,
                min_conf: 0.5,
                min_lift: 1.0,
                target: 'OnTime-EVENT-Delivery',
            },
        },
        data: { objective: { name: 'itemset', parameters: { min_sup: 0.2 } } },
        attributes: [],
        init() {
            this.attributes = window.initialItemAttributes.map((a) => ({
                attribute: a.attribute,
                type: a.type,
                qualifier: a.qualifier,
            }));
            this.$watch('data.objective.name', (newName) => {
                this.data.objective.parameters = JSON.parse(
                    JSON.stringify(this.defaultParameters[newName])
                );
            });
        },
        submitForm() {
            fetch(this.$el.action, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.data),
            })
                .then((r) => r.json())
                .then((json) => {
                    if (json.status === 'success' && json.redirect_url) {
                        window.location.href = json.redirect_url;
                    } else {
                        alert('Fehler beim Verarbeiten!');
                    }
                });
        },
    }));

    document.addEventListener('DOMContentLoaded', () => {
        const btn = document.getElementById('scrollToTopBtn');
        if (btn) {
            window.addEventListener('scroll', () => {
                btn.style.display = window.scrollY > 20 ? 'block' : 'none';
            });
            btn.addEventListener('click', () => {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }
        document.querySelectorAll('.support-fill, .confidence-fill').forEach((bar) => {
            const w = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => (bar.style.width = w), 100);
        });
    });

    const pattern = document.getElementById('initial-pattern');
    window.initialFrequentItemsets = JSON.parse(pattern.textContent);
    window.initialRules = JSON.parse(pattern.textContent);

    Alpine.data('itemsetFilter', () => ({
        sets: [],
        inputDsl: '',
        dsl: '',
        init() {
            this.sets = window.initialFrequentItemsets || [];
        },
        applyFilter() {
            this.dsl = this.inputDsl.trim();
        },
        resetFilter() {
            this.inputDsl = '';
            this.dsl = '';
        },
        splitClauses(str) {
            const parts = [],
                chars = [...str];
            let depth = 0,
                buf = '';
            for (let c of chars) {
                if (c === '(') depth++;
                if (c === ')') depth--;
                if (c === ',' && depth === 0) {
                    parts.push(buf.trim());
                    buf = '';
                } else buf += c;
            }
            if (buf.trim()) parts.push(buf.trim());
            return parts;
        },
        checkItemset(set, expr) {
            const tokens = expr.split(/\s*,\s*/).map((t) => {
                const [a, v] = t.split(':').map((s) => s.trim());
                return { attr: a, val: v };
            });
            return tokens.every(({ attr, val }) =>
                set.item.some(
                    (i) =>
                        (attr === '*' || i.attribute === attr) && (val === '*' || i.value === val)
                )
            );
        },
        checkSupport(set, expr) {
            let [min, max] = expr.split(':').map((s) => s.trim());
            min = min === '' ? 0 : parseFloat(min);
            max = max === '' ? Infinity : parseFloat(max);
            return set.support >= min && set.support <= max;
        },
        evaluate(expr, set) {
            expr = expr.trim();
            if (!expr) return true;
            const m = expr.match(/^(AND|OR)\(([\s\S]*)\)$/i);
            if (m) {
                const op = m[1],
                    parts = this.splitClauses(m[2]);
                return op === 'OR'
                    ? parts.some((p) => this.evaluate(p, set))
                    : parts.every((p) => this.evaluate(p, set));
            }
            if (expr.startsWith('Itemset(')) return this.checkItemset(set, expr.slice(8, -1));
            if (expr.startsWith('Support(')) return this.checkSupport(set, expr.slice(8, -1));
            return true;
        },
        get filtered() {
            return this.dsl ? this.sets.filter((s) => this.evaluate(this.dsl, s)) : this.sets;
        },
        get filteredMaxSupport() {
            return this.filtered.length ? Math.max(...this.filtered.map((s) => s.support)) : 0;
        },
        get filteredMinSupport() {
            return this.filtered.length ? Math.min(...this.filtered.map((s) => s.support)) : 0;
        },
        get filteredAvgSupport() {
            return this.filtered.length
                ? this.filtered.reduce((a, b) => a + b.support, 0) / this.filtered.length
                : 0;
        },
    }));

    Alpine.data('ruleFilter', () => ({
        rules: [],
        inputDsl: '',
        dsl: '',
        init() {
            this.rules = window.initialRules || [];
        },
        applyFilter() {
            this.dsl = this.inputDsl.trim();
        },
        resetFilter() {
            this.inputDsl = '';
            this.dsl = '';
        },
        splitClauses(str) {
            const res = [];
            let buf = '',
                depth = 0;
            for (let ch of str) {
                if (ch === '(') depth++;
                if (ch === ')') depth--;
                if (ch === ',' && depth === 0) {
                    res.push(buf.trim());
                    buf = '';
                } else buf += ch;
            }
            if (buf.trim()) res.push(buf.trim());
            return res;
        },
        checkNumeric(rule, field, expr) {
            let [min, max] = expr.split(':').map((s) => s.trim());
            min = min === '' ? 0 : parseFloat(min);
            max = max === '' ? Infinity : parseFloat(max);
            const val = rule[field];
            return val >= min && val <= max;
        },
        checkSide(arr, expr) {
            if (!expr) return true;
            return expr.split(',').every((tok) => {
                let [a, v] = tok.split(':').map((s) => s.replace(/"/g, '').trim());
                return arr.some(
                    (item) => (a === '*' || item.attribute === a) && (v === '*' || item.value === v)
                );
            });
        },
        evaluate(expr, rule) {
            expr = expr.trim();
            if (!expr) return true;
            const m = expr.match(/^(AND|OR)\(([\s\S]*)\)$/i);
            if (m) {
                const op = m[1].toUpperCase(),
                    parts = this.splitClauses(m[2]);
                return op === 'OR'
                    ? parts.some((p) => this.evaluate(p, rule))
                    : parts.every((p) => this.evaluate(p, rule));
            }
            if (expr.startsWith('Support('))
                return this.checkNumeric(rule, 'support', expr.slice(8, -1));
            if (expr.startsWith('Confidence('))
                return this.checkNumeric(rule, 'confidence', expr.slice(11, -1));
            if (expr.startsWith('Lift(')) return this.checkNumeric(rule, 'lift', expr.slice(5, -1));
            if (expr.startsWith('Antecedent('))
                return this.checkSide(rule.antecedents, expr.slice(11, -1));
            if (expr.startsWith('Consequent('))
                return this.checkSide(rule.consequents, expr.slice(11, -1));
            return true;
        },
        get filteredRules() {
            return this.dsl ? this.rules.filter((r) => this.evaluate(this.dsl, r)) : this.rules;
        },
    }));
});
