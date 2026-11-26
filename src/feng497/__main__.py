"""python -m feng497 ve feng497 terminal komutlarının giriş noktası."""

import argparse
from importlib import import_module

COMMANDS = {
    "inspect": ("inspect_data", "Eğitim verisini ve hedef sütununu incele."),
    "plot": ("plot_data", "Eğitim verisinin grafiklerini reports/ altına yaz."),
    "baseline": ("baseline", "Ortalama ve doğrusal regresyon referanslarını karşılaştır."),
    "random-forest": ("random_forest", "Random Forest modelini çapraz doğrulamayla karşılaştır."),
    "gpr": ("gpr", "GPR modellerini aynı doğrulama bölmeleriyle karşılaştır."),
    "train": ("train", "Son modeli eğit, ayrılmış testleri değerlendir ve models/ altına kaydet."),
    "predict": ("predict", "Kaydedilmiş modelle yeni bir geometri için tahmin al."),
    "inspect-composite": ("composite", "Kompozit verisini ve ayrılmış grupları incele."),
    "train-composite": ("train_composite", "Kompozitte ilk hasar yükü modelini eğit ve test et."),
    "predict-composite": ("predict_composite", "Fiber açısıyla kompozitte ilk hasar yükünü tahmin et."),
    "train-fatigue": ("train_fatigue", "İlk yorulma çatlağı modelini S–N verisiyle eğit ve test et."),
    "predict-fatigue": ("predict_fatigue", "Verilen tekrarlı yükte ilk çatlağa kadar çevrim sayısını tahmin et."),
}
COMMANDS_WITH_ARGS = {"predict", "predict-composite", "train-fatigue", "predict-fatigue"}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="feng497",
        description="Delikli plakada çelik gerilmesi, kompozit ilk hasar ve yorulma ömrü tahmini.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(
            name, help=description, description=description,
            add_help=name not in COMMANDS_WITH_ARGS,
        )
    args, remaining = parser.parse_known_args(argv)
    if args.command not in COMMANDS_WITH_ARGS and remaining:
        parser.error("Tanınmayan argümanlar: " + " ".join(remaining))
    module = import_module("." + COMMANDS[args.command][0], package="feng497")
    if args.command in COMMANDS_WITH_ARGS:
        return module.main(remaining)
    return module.main()


if __name__ == "__main__":
    main()
