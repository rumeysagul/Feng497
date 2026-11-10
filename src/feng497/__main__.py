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
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="feng497",
        description="Delikli plakada çelik gerilmesi ve kompozit ilk hasar yükü tahmini.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        # predict kendi fiziksel girdi seçeneklerini ve --help çıktısını yönetir.
        commands.add_parser(
            name, help=description, description=description,
            add_help=name not in {"predict", "predict-composite"}
        )
    args, remaining = parser.parse_known_args(argv)
    if args.command not in {"predict", "predict-composite"} and remaining:
        parser.error("Tanınmayan argümanlar: " + " ".join(remaining))
    module = import_module("." + COMMANDS[args.command][0], package="feng497")
    if args.command in {"predict", "predict-composite"}:
        return module.main(remaining)
    return module.main()


if __name__ == "__main__":
    main()
