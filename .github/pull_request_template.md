## O que mudou
<!-- Resumo do PR e qual PR do plano ele implementa (docs/plano_desenvolvimento.md) -->

## Como testar
```bash
# comandos para reproduzir
```

## Evidências
<!-- logs, screenshots, trechos de JSON de relatórios/alertas (sem PII, chaves mascaradas) -->

## Definition of Done global
- [ ] Branch atualizada com `main`, sem conflitos
- [ ] `ruff check`, `ruff format --check` e `mypy src` limpos
- [ ] `pytest -m "not azure and not slow"` verde; cobertura do código novo ≥ 70%
- [ ] Nenhum segredo commitado; `.env.example` atualizado com novas variáveis
- [ ] Nenhum dado bruto, peso de modelo ou arquivo > 5 MB (exceto `data/samples/`)
- [ ] Novas dependências no `requirements.txt` com piso e teto no próximo major, e comentário quando o motivo não for óbvio
- [ ] Docstrings em funções públicas; parâmetros em `configs/*.yaml`
- [ ] Logs sem PII (`patient_id` pseudonimizado)
- [ ] README atualizado (seção do módulo e comando de execução)
- [ ] Checklist específico do PR (docs/plano_desenvolvimento.md) concluído
